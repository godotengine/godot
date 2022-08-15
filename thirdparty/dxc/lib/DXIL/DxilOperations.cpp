///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilOperations.cpp                                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Implementation of DXIL operation tables.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilOperations.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilInstructions.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
using std::vector;
using std::string;


namespace hlsl {

using OC = OP::OpCode;
using OCC = OP::OpCodeClass;

//------------------------------------------------------------------------------
//
//  OP class const-static data and related static methods.
//
/* <py>
import hctdb_instrhelp
</py> */
/* <py::lines('OPCODE-OLOADS')>hctdb_instrhelp.get_oloads_props()</py>*/
// OPCODE-OLOADS:BEGIN
const OP::OpCodeProperty OP::m_OpCodeProps[(unsigned)OP::OpCode::NumOpCodes] = {
//   OpCode                       OpCode name,                OpCodeClass                    OpCodeClass name,              void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj,  function attribute
  // Temporary, indexable, input, output registers                                                                           void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::TempRegLoad,             "TempRegLoad",              OCC::TempRegLoad,              "tempRegLoad",               { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::TempRegStore,            "TempRegStore",             OCC::TempRegStore,             "tempRegStore",              { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::None,     },
  {  OC::MinPrecXRegLoad,         "MinPrecXRegLoad",          OCC::MinPrecXRegLoad,          "minPrecXRegLoad",           { false,  true, false, false, false, false,  true, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::MinPrecXRegStore,        "MinPrecXRegStore",         OCC::MinPrecXRegStore,         "minPrecXRegStore",          { false,  true, false, false, false, false,  true, false, false, false, false}, Attribute::None,     },
  {  OC::LoadInput,               "LoadInput",                OCC::LoadInput,                "loadInput",                 { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::StoreOutput,             "StoreOutput",              OCC::StoreOutput,              "storeOutput",               { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::None,     },

  // Unary float                                                                                                             void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::FAbs,                    "FAbs",                     OCC::Unary,                    "unary",                     { false,  true,  true,  true, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Saturate,                "Saturate",                 OCC::Unary,                    "unary",                     { false,  true,  true,  true, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::IsNaN,                   "IsNaN",                    OCC::IsSpecialFloat,           "isSpecialFloat",            { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::IsInf,                   "IsInf",                    OCC::IsSpecialFloat,           "isSpecialFloat",            { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::IsFinite,                "IsFinite",                 OCC::IsSpecialFloat,           "isSpecialFloat",            { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::IsNormal,                "IsNormal",                 OCC::IsSpecialFloat,           "isSpecialFloat",            { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Cos,                     "Cos",                      OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Sin,                     "Sin",                      OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Tan,                     "Tan",                      OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Acos,                    "Acos",                     OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Asin,                    "Asin",                     OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Atan,                    "Atan",                     OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Hcos,                    "Hcos",                     OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Hsin,                    "Hsin",                     OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Htan,                    "Htan",                     OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Exp,                     "Exp",                      OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Frc,                     "Frc",                      OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Log,                     "Log",                      OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Sqrt,                    "Sqrt",                     OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Rsqrt,                   "Rsqrt",                    OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Unary float - rounding                                                                                                  void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Round_ne,                "Round_ne",                 OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Round_ni,                "Round_ni",                 OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Round_pi,                "Round_pi",                 OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Round_z,                 "Round_z",                  OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Unary int                                                                                                               void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Bfrev,                   "Bfrev",                    OCC::Unary,                    "unary",                     { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },
  {  OC::Countbits,               "Countbits",                OCC::UnaryBits,                "unaryBits",                 { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },
  {  OC::FirstbitLo,              "FirstbitLo",               OCC::UnaryBits,                "unaryBits",                 { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },

  // Unary uint                                                                                                              void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::FirstbitHi,              "FirstbitHi",               OCC::UnaryBits,                "unaryBits",                 { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },

  // Unary int                                                                                                               void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::FirstbitSHi,             "FirstbitSHi",              OCC::UnaryBits,                "unaryBits",                 { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },

  // Binary float                                                                                                            void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::FMax,                    "FMax",                     OCC::Binary,                   "binary",                    { false,  true,  true,  true, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::FMin,                    "FMin",                     OCC::Binary,                   "binary",                    { false,  true,  true,  true, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Binary int                                                                                                              void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::IMax,                    "IMax",                     OCC::Binary,                   "binary",                    { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },
  {  OC::IMin,                    "IMin",                     OCC::Binary,                   "binary",                    { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },

  // Binary uint                                                                                                             void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::UMax,                    "UMax",                     OCC::Binary,                   "binary",                    { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },
  {  OC::UMin,                    "UMin",                     OCC::Binary,                   "binary",                    { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },

  // Binary int with two outputs                                                                                             void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::IMul,                    "IMul",                     OCC::BinaryWithTwoOuts,        "binaryWithTwoOuts",         { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Binary uint with two outputs                                                                                            void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::UMul,                    "UMul",                     OCC::BinaryWithTwoOuts,        "binaryWithTwoOuts",         { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::UDiv,                    "UDiv",                     OCC::BinaryWithTwoOuts,        "binaryWithTwoOuts",         { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Binary uint with carry or borrow                                                                                        void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::UAddc,                   "UAddc",                    OCC::BinaryWithCarryOrBorrow,  "binaryWithCarryOrBorrow",   { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::USubb,                   "USubb",                    OCC::BinaryWithCarryOrBorrow,  "binaryWithCarryOrBorrow",   { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Tertiary float                                                                                                          void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::FMad,                    "FMad",                     OCC::Tertiary,                 "tertiary",                  { false,  true,  true,  true, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Fma,                     "Fma",                      OCC::Tertiary,                 "tertiary",                  { false, false, false,  true, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Tertiary int                                                                                                            void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::IMad,                    "IMad",                     OCC::Tertiary,                 "tertiary",                  { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },

  // Tertiary uint                                                                                                           void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::UMad,                    "UMad",                     OCC::Tertiary,                 "tertiary",                  { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadNone, },

  // Tertiary int                                                                                                            void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Msad,                    "Msad",                     OCC::Tertiary,                 "tertiary",                  { false, false, false, false, false, false, false,  true,  true, false, false}, Attribute::ReadNone, },
  {  OC::Ibfe,                    "Ibfe",                     OCC::Tertiary,                 "tertiary",                  { false, false, false, false, false, false, false,  true,  true, false, false}, Attribute::ReadNone, },

  // Tertiary uint                                                                                                           void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Ubfe,                    "Ubfe",                     OCC::Tertiary,                 "tertiary",                  { false, false, false, false, false, false, false,  true,  true, false, false}, Attribute::ReadNone, },

  // Quaternary                                                                                                              void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Bfi,                     "Bfi",                      OCC::Quaternary,               "quaternary",                { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Dot                                                                                                                     void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Dot2,                    "Dot2",                     OCC::Dot2,                     "dot2",                      { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Dot3,                    "Dot3",                     OCC::Dot3,                     "dot3",                      { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Dot4,                    "Dot4",                     OCC::Dot4,                     "dot4",                      { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Resources                                                                                                               void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::CreateHandle,            "CreateHandle",             OCC::CreateHandle,             "createHandle",              {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::CBufferLoad,             "CBufferLoad",              OCC::CBufferLoad,              "cbufferLoad",               { false,  true,  true,  true, false,  true,  true,  true,  true, false, false}, Attribute::ReadOnly, },
  {  OC::CBufferLoadLegacy,       "CBufferLoadLegacy",        OCC::CBufferLoadLegacy,        "cbufferLoadLegacy",         { false,  true,  true,  true, false, false,  true,  true,  true, false, false}, Attribute::ReadOnly, },

  // Resources - sample                                                                                                      void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Sample,                  "Sample",                   OCC::Sample,                   "sample",                    { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::SampleBias,              "SampleBias",               OCC::SampleBias,               "sampleBias",                { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::SampleLevel,             "SampleLevel",              OCC::SampleLevel,              "sampleLevel",               { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::SampleGrad,              "SampleGrad",               OCC::SampleGrad,               "sampleGrad",                { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::SampleCmp,               "SampleCmp",                OCC::SampleCmp,                "sampleCmp",                 { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::SampleCmpLevelZero,      "SampleCmpLevelZero",       OCC::SampleCmpLevelZero,       "sampleCmpLevelZero",        { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },

  // Resources                                                                                                               void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::TextureLoad,             "TextureLoad",              OCC::TextureLoad,              "textureLoad",               { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::TextureStore,            "TextureStore",             OCC::TextureStore,             "textureStore",              { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::None,     },
  {  OC::BufferLoad,              "BufferLoad",               OCC::BufferLoad,               "bufferLoad",                { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::BufferStore,             "BufferStore",              OCC::BufferStore,              "bufferStore",               { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::None,     },
  {  OC::BufferUpdateCounter,     "BufferUpdateCounter",      OCC::BufferUpdateCounter,      "bufferUpdateCounter",       {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::CheckAccessFullyMapped,  "CheckAccessFullyMapped",   OCC::CheckAccessFullyMapped,   "checkAccessFullyMapped",    { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::GetDimensions,           "GetDimensions",            OCC::GetDimensions,            "getDimensions",             {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },

  // Resources - gather                                                                                                      void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::TextureGather,           "TextureGather",            OCC::TextureGather,            "textureGather",             { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::TextureGatherCmp,        "TextureGatherCmp",         OCC::TextureGatherCmp,         "textureGatherCmp",          { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadOnly, },

  // Resources - sample                                                                                                      void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Texture2DMSGetSamplePosition, "Texture2DMSGetSamplePosition", OCC::Texture2DMSGetSamplePosition, "texture2DMSGetSamplePosition", {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RenderTargetGetSamplePosition, "RenderTargetGetSamplePosition", OCC::RenderTargetGetSamplePosition, "renderTargetGetSamplePosition", {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RenderTargetGetSampleCount, "RenderTargetGetSampleCount", OCC::RenderTargetGetSampleCount, "renderTargetGetSampleCount", {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },

  // Synchronization                                                                                                         void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::AtomicBinOp,             "AtomicBinOp",              OCC::AtomicBinOp,              "atomicBinOp",               { false, false, false, false, false, false, false,  true,  true, false, false}, Attribute::None,     },
  {  OC::AtomicCompareExchange,   "AtomicCompareExchange",    OCC::AtomicCompareExchange,    "atomicCompareExchange",     { false, false, false, false, false, false, false,  true,  true, false, false}, Attribute::None,     },
  {  OC::Barrier,                 "Barrier",                  OCC::Barrier,                  "barrier",                   {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::NoDuplicate, },

  // Derivatives                                                                                                             void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::CalculateLOD,            "CalculateLOD",             OCC::CalculateLOD,             "calculateLOD",              { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },

  // Pixel shader                                                                                                            void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Discard,                 "Discard",                  OCC::Discard,                  "discard",                   {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },

  // Derivatives                                                                                                             void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::DerivCoarseX,            "DerivCoarseX",             OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::DerivCoarseY,            "DerivCoarseY",             OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::DerivFineX,              "DerivFineX",               OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::DerivFineY,              "DerivFineY",               OCC::Unary,                    "unary",                     { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Pixel shader                                                                                                            void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::EvalSnapped,             "EvalSnapped",              OCC::EvalSnapped,              "evalSnapped",               { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::EvalSampleIndex,         "EvalSampleIndex",          OCC::EvalSampleIndex,          "evalSampleIndex",           { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::EvalCentroid,            "EvalCentroid",             OCC::EvalCentroid,             "evalCentroid",              { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::SampleIndex,             "SampleIndex",              OCC::SampleIndex,              "sampleIndex",               { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::Coverage,                "Coverage",                 OCC::Coverage,                 "coverage",                  { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::InnerCoverage,           "InnerCoverage",            OCC::InnerCoverage,            "innerCoverage",             { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Compute/Mesh/Amplification shader                                                                                       void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::ThreadId,                "ThreadId",                 OCC::ThreadId,                 "threadId",                  { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::GroupId,                 "GroupId",                  OCC::GroupId,                  "groupId",                   { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::ThreadIdInGroup,         "ThreadIdInGroup",          OCC::ThreadIdInGroup,          "threadIdInGroup",           { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::FlattenedThreadIdInGroup, "FlattenedThreadIdInGroup", OCC::FlattenedThreadIdInGroup, "flattenedThreadIdInGroup",  { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Geometry shader                                                                                                         void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::EmitStream,              "EmitStream",               OCC::EmitStream,               "emitStream",                {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::CutStream,               "CutStream",                OCC::CutStream,                "cutStream",                 {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::EmitThenCutStream,       "EmitThenCutStream",        OCC::EmitThenCutStream,        "emitThenCutStream",         {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::GSInstanceID,            "GSInstanceID",             OCC::GSInstanceID,             "gsInstanceID",              { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Double precision                                                                                                        void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::MakeDouble,              "MakeDouble",               OCC::MakeDouble,               "makeDouble",                { false, false, false,  true, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::SplitDouble,             "SplitDouble",              OCC::SplitDouble,              "splitDouble",               { false, false, false,  true, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Domain and hull shader                                                                                                  void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::LoadOutputControlPoint,  "LoadOutputControlPoint",   OCC::LoadOutputControlPoint,   "loadOutputControlPoint",    { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::LoadPatchConstant,       "LoadPatchConstant",        OCC::LoadPatchConstant,        "loadPatchConstant",         { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadNone, },

  // Domain shader                                                                                                           void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::DomainLocation,          "DomainLocation",           OCC::DomainLocation,           "domainLocation",            { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Hull shader                                                                                                             void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::StorePatchConstant,      "StorePatchConstant",       OCC::StorePatchConstant,       "storePatchConstant",        { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::None,     },
  {  OC::OutputControlPointID,    "OutputControlPointID",     OCC::OutputControlPointID,     "outputControlPointID",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Hull, Domain and Geometry shaders                                                                                       void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::PrimitiveID,             "PrimitiveID",              OCC::PrimitiveID,              "primitiveID",               { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Other                                                                                                                   void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::CycleCounterLegacy,      "CycleCounterLegacy",       OCC::CycleCounterLegacy,       "cycleCounterLegacy",        {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },

  // Wave                                                                                                                    void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::WaveIsFirstLane,         "WaveIsFirstLane",          OCC::WaveIsFirstLane,          "waveIsFirstLane",           {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::WaveGetLaneIndex,        "WaveGetLaneIndex",         OCC::WaveGetLaneIndex,         "waveGetLaneIndex",          {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::WaveGetLaneCount,        "WaveGetLaneCount",         OCC::WaveGetLaneCount,         "waveGetLaneCount",          {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::WaveAnyTrue,             "WaveAnyTrue",              OCC::WaveAnyTrue,              "waveAnyTrue",               {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::WaveAllTrue,             "WaveAllTrue",              OCC::WaveAllTrue,              "waveAllTrue",               {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::WaveActiveAllEqual,      "WaveActiveAllEqual",       OCC::WaveActiveAllEqual,       "waveActiveAllEqual",        { false,  true,  true,  true,  true,  true,  true,  true,  true, false, false}, Attribute::None,     },
  {  OC::WaveActiveBallot,        "WaveActiveBallot",         OCC::WaveActiveBallot,         "waveActiveBallot",          {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::WaveReadLaneAt,          "WaveReadLaneAt",           OCC::WaveReadLaneAt,           "waveReadLaneAt",            { false,  true,  true,  true,  true,  true,  true,  true,  true, false, false}, Attribute::None,     },
  {  OC::WaveReadLaneFirst,       "WaveReadLaneFirst",        OCC::WaveReadLaneFirst,        "waveReadLaneFirst",         { false,  true,  true,  true,  true,  true,  true,  true,  true, false, false}, Attribute::None,     },
  {  OC::WaveActiveOp,            "WaveActiveOp",             OCC::WaveActiveOp,             "waveActiveOp",              { false,  true,  true,  true,  true,  true,  true,  true,  true, false, false}, Attribute::None,     },
  {  OC::WaveActiveBit,           "WaveActiveBit",            OCC::WaveActiveBit,            "waveActiveBit",             { false, false, false, false, false,  true,  true,  true,  true, false, false}, Attribute::None,     },
  {  OC::WavePrefixOp,            "WavePrefixOp",             OCC::WavePrefixOp,             "wavePrefixOp",              { false,  true,  true,  true, false,  true,  true,  true,  true, false, false}, Attribute::None,     },

  // Quad Wave Ops                                                                                                           void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::QuadReadLaneAt,          "QuadReadLaneAt",           OCC::QuadReadLaneAt,           "quadReadLaneAt",            { false,  true,  true,  true,  true,  true,  true,  true,  true, false, false}, Attribute::None,     },
  {  OC::QuadOp,                  "QuadOp",                   OCC::QuadOp,                   "quadOp",                    { false,  true,  true,  true, false,  true,  true,  true,  true, false, false}, Attribute::None,     },

  // Bitcasts with different sizes                                                                                           void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::BitcastI16toF16,         "BitcastI16toF16",          OCC::BitcastI16toF16,          "bitcastI16toF16",           {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::BitcastF16toI16,         "BitcastF16toI16",          OCC::BitcastF16toI16,          "bitcastF16toI16",           {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::BitcastI32toF32,         "BitcastI32toF32",          OCC::BitcastI32toF32,          "bitcastI32toF32",           {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::BitcastF32toI32,         "BitcastF32toI32",          OCC::BitcastF32toI32,          "bitcastF32toI32",           {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::BitcastI64toF64,         "BitcastI64toF64",          OCC::BitcastI64toF64,          "bitcastI64toF64",           {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::BitcastF64toI64,         "BitcastF64toI64",          OCC::BitcastF64toI64,          "bitcastF64toI64",           {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Legacy floating-point                                                                                                   void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::LegacyF32ToF16,          "LegacyF32ToF16",           OCC::LegacyF32ToF16,           "legacyF32ToF16",            {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::LegacyF16ToF32,          "LegacyF16ToF32",           OCC::LegacyF16ToF32,           "legacyF16ToF32",            {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Double precision                                                                                                        void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::LegacyDoubleToFloat,     "LegacyDoubleToFloat",      OCC::LegacyDoubleToFloat,      "legacyDoubleToFloat",       {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::LegacyDoubleToSInt32,    "LegacyDoubleToSInt32",     OCC::LegacyDoubleToSInt32,     "legacyDoubleToSInt32",      {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::LegacyDoubleToUInt32,    "LegacyDoubleToUInt32",     OCC::LegacyDoubleToUInt32,     "legacyDoubleToUInt32",      {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Wave                                                                                                                    void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::WaveAllBitCount,         "WaveAllBitCount",          OCC::WaveAllOp,                "waveAllOp",                 {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::WavePrefixBitCount,      "WavePrefixBitCount",       OCC::WavePrefixOp,             "wavePrefixOp",              {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },

  // Pixel shader                                                                                                            void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::AttributeAtVertex,       "AttributeAtVertex",        OCC::AttributeAtVertex,        "attributeAtVertex",         { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::ReadNone, },

  // Graphics shader                                                                                                         void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::ViewID,                  "ViewID",                   OCC::ViewID,                   "viewID",                    { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Resources                                                                                                               void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::RawBufferLoad,           "RawBufferLoad",            OCC::RawBufferLoad,            "rawBufferLoad",             { false,  true,  true,  true, false, false,  true,  true,  true, false, false}, Attribute::ReadOnly, },
  {  OC::RawBufferStore,          "RawBufferStore",           OCC::RawBufferStore,           "rawBufferStore",            { false,  true,  true,  true, false, false,  true,  true,  true, false, false}, Attribute::None,     },

  // Raytracing object space uint System Values                                                                              void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::InstanceID,              "InstanceID",               OCC::InstanceID,               "instanceID",                { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::InstanceIndex,           "InstanceIndex",            OCC::InstanceIndex,            "instanceIndex",             { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Raytracing hit uint System Values                                                                                       void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::HitKind,                 "HitKind",                  OCC::HitKind,                  "hitKind",                   { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Raytracing uint System Values                                                                                           void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::RayFlags,                "RayFlags",                 OCC::RayFlags,                 "rayFlags",                  { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Ray Dispatch Arguments                                                                                                  void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::DispatchRaysIndex,       "DispatchRaysIndex",        OCC::DispatchRaysIndex,        "dispatchRaysIndex",         { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::DispatchRaysDimensions,  "DispatchRaysDimensions",   OCC::DispatchRaysDimensions,   "dispatchRaysDimensions",    { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Ray Vectors                                                                                                             void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::WorldRayOrigin,          "WorldRayOrigin",           OCC::WorldRayOrigin,           "worldRayOrigin",            { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::WorldRayDirection,       "WorldRayDirection",        OCC::WorldRayDirection,        "worldRayDirection",         { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Ray object space Vectors                                                                                                void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::ObjectRayOrigin,         "ObjectRayOrigin",          OCC::ObjectRayOrigin,          "objectRayOrigin",           { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::ObjectRayDirection,      "ObjectRayDirection",       OCC::ObjectRayDirection,       "objectRayDirection",        { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Ray Transforms                                                                                                          void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::ObjectToWorld,           "ObjectToWorld",            OCC::ObjectToWorld,            "objectToWorld",             { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::WorldToObject,           "WorldToObject",            OCC::WorldToObject,            "worldToObject",             { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // RayT                                                                                                                    void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::RayTMin,                 "RayTMin",                  OCC::RayTMin,                  "rayTMin",                   { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::RayTCurrent,             "RayTCurrent",              OCC::RayTCurrent,              "rayTCurrent",               { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },

  // AnyHit Terminals                                                                                                        void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::IgnoreHit,               "IgnoreHit",                OCC::IgnoreHit,                "ignoreHit",                 {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::NoReturn, },
  {  OC::AcceptHitAndEndSearch,   "AcceptHitAndEndSearch",    OCC::AcceptHitAndEndSearch,    "acceptHitAndEndSearch",     {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::NoReturn, },

  // Indirect Shader Invocation                                                                                              void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::TraceRay,                "TraceRay",                 OCC::TraceRay,                 "traceRay",                  { false, false, false, false, false, false, false, false, false,  true, false}, Attribute::None,     },
  {  OC::ReportHit,               "ReportHit",                OCC::ReportHit,                "reportHit",                 { false, false, false, false, false, false, false, false, false,  true, false}, Attribute::None,     },
  {  OC::CallShader,              "CallShader",               OCC::CallShader,               "callShader",                { false, false, false, false, false, false, false, false, false,  true, false}, Attribute::None,     },

  // Library create handle from resource struct (like HL intrinsic)                                                          void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::CreateHandleForLib,      "CreateHandleForLib",       OCC::CreateHandleForLib,       "createHandleForLib",        { false, false, false, false, false, false, false, false, false, false,  true}, Attribute::ReadOnly, },

  // Raytracing object space uint System Values                                                                              void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::PrimitiveIndex,          "PrimitiveIndex",           OCC::PrimitiveIndex,           "primitiveIndex",            { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Dot product with accumulate                                                                                             void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Dot2AddHalf,             "Dot2AddHalf",              OCC::Dot2AddHalf,              "dot2AddHalf",               { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::Dot4AddI8Packed,         "Dot4AddI8Packed",          OCC::Dot4AddPacked,            "dot4AddPacked",             { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },
  {  OC::Dot4AddU8Packed,         "Dot4AddU8Packed",          OCC::Dot4AddPacked,            "dot4AddPacked",             { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Wave                                                                                                                    void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::WaveMatch,               "WaveMatch",                OCC::WaveMatch,                "waveMatch",                 { false,  true,  true,  true, false,  true,  true,  true,  true, false, false}, Attribute::None,     },
  {  OC::WaveMultiPrefixOp,       "WaveMultiPrefixOp",        OCC::WaveMultiPrefixOp,        "waveMultiPrefixOp",         { false,  true,  true,  true, false,  true,  true,  true,  true, false, false}, Attribute::None,     },
  {  OC::WaveMultiPrefixBitCount, "WaveMultiPrefixBitCount",  OCC::WaveMultiPrefixBitCount,  "waveMultiPrefixBitCount",   {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },

  // Mesh shader instructions                                                                                                void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::SetMeshOutputCounts,     "SetMeshOutputCounts",      OCC::SetMeshOutputCounts,      "setMeshOutputCounts",       {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::EmitIndices,             "EmitIndices",              OCC::EmitIndices,              "emitIndices",               {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::GetMeshPayload,          "GetMeshPayload",           OCC::GetMeshPayload,           "getMeshPayload",            { false, false, false, false, false, false, false, false, false,  true, false}, Attribute::ReadOnly, },
  {  OC::StoreVertexOutput,       "StoreVertexOutput",        OCC::StoreVertexOutput,        "storeVertexOutput",         { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::None,     },
  {  OC::StorePrimitiveOutput,    "StorePrimitiveOutput",     OCC::StorePrimitiveOutput,     "storePrimitiveOutput",      { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::None,     },

  // Amplification shader instructions                                                                                       void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::DispatchMesh,            "DispatchMesh",             OCC::DispatchMesh,             "dispatchMesh",              { false, false, false, false, false, false, false, false, false,  true, false}, Attribute::None,     },

  // Sampler Feedback                                                                                                        void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::WriteSamplerFeedback,    "WriteSamplerFeedback",     OCC::WriteSamplerFeedback,     "writeSamplerFeedback",      {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::WriteSamplerFeedbackBias, "WriteSamplerFeedbackBias", OCC::WriteSamplerFeedbackBias, "writeSamplerFeedbackBias",  {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::WriteSamplerFeedbackLevel, "WriteSamplerFeedbackLevel", OCC::WriteSamplerFeedbackLevel, "writeSamplerFeedbackLevel", {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::WriteSamplerFeedbackGrad, "WriteSamplerFeedbackGrad", OCC::WriteSamplerFeedbackGrad, "writeSamplerFeedbackGrad",  {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },

  // Inline Ray Query                                                                                                        void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::AllocateRayQuery,        "AllocateRayQuery",         OCC::AllocateRayQuery,         "allocateRayQuery",          {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::RayQuery_TraceRayInline, "RayQuery_TraceRayInline",  OCC::RayQuery_TraceRayInline,  "rayQuery_TraceRayInline",   {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::RayQuery_Proceed,        "RayQuery_Proceed",         OCC::RayQuery_Proceed,         "rayQuery_Proceed",          { false, false, false, false,  true, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::RayQuery_Abort,          "RayQuery_Abort",           OCC::RayQuery_Abort,           "rayQuery_Abort",            {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::RayQuery_CommitNonOpaqueTriangleHit, "RayQuery_CommitNonOpaqueTriangleHit", OCC::RayQuery_CommitNonOpaqueTriangleHit, "rayQuery_CommitNonOpaqueTriangleHit", {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::RayQuery_CommitProceduralPrimitiveHit, "RayQuery_CommitProceduralPrimitiveHit", OCC::RayQuery_CommitProceduralPrimitiveHit, "rayQuery_CommitProceduralPrimitiveHit", {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::None,     },
  {  OC::RayQuery_CommittedStatus, "RayQuery_CommittedStatus", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateType,  "RayQuery_CandidateType",   OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateObjectToWorld3x4, "RayQuery_CandidateObjectToWorld3x4", OCC::RayQuery_StateMatrix,     "rayQuery_StateMatrix",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateWorldToObject3x4, "RayQuery_CandidateWorldToObject3x4", OCC::RayQuery_StateMatrix,     "rayQuery_StateMatrix",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedObjectToWorld3x4, "RayQuery_CommittedObjectToWorld3x4", OCC::RayQuery_StateMatrix,     "rayQuery_StateMatrix",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedWorldToObject3x4, "RayQuery_CommittedWorldToObject3x4", OCC::RayQuery_StateMatrix,     "rayQuery_StateMatrix",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateProceduralPrimitiveNonOpaque, "RayQuery_CandidateProceduralPrimitiveNonOpaque", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false,  true, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateTriangleFrontFace, "RayQuery_CandidateTriangleFrontFace", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false,  true, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedTriangleFrontFace, "RayQuery_CommittedTriangleFrontFace", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false,  true, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateTriangleBarycentrics, "RayQuery_CandidateTriangleBarycentrics", OCC::RayQuery_StateVector,     "rayQuery_StateVector",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedTriangleBarycentrics, "RayQuery_CommittedTriangleBarycentrics", OCC::RayQuery_StateVector,     "rayQuery_StateVector",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_RayFlags,       "RayQuery_RayFlags",        OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_WorldRayOrigin, "RayQuery_WorldRayOrigin",  OCC::RayQuery_StateVector,     "rayQuery_StateVector",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_WorldRayDirection, "RayQuery_WorldRayDirection", OCC::RayQuery_StateVector,     "rayQuery_StateVector",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_RayTMin,        "RayQuery_RayTMin",         OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateTriangleRayT, "RayQuery_CandidateTriangleRayT", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedRayT,  "RayQuery_CommittedRayT",   OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateInstanceIndex, "RayQuery_CandidateInstanceIndex", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateInstanceID, "RayQuery_CandidateInstanceID", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateGeometryIndex, "RayQuery_CandidateGeometryIndex", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidatePrimitiveIndex, "RayQuery_CandidatePrimitiveIndex", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateObjectRayOrigin, "RayQuery_CandidateObjectRayOrigin", OCC::RayQuery_StateVector,     "rayQuery_StateVector",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CandidateObjectRayDirection, "RayQuery_CandidateObjectRayDirection", OCC::RayQuery_StateVector,     "rayQuery_StateVector",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedInstanceIndex, "RayQuery_CommittedInstanceIndex", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedInstanceID, "RayQuery_CommittedInstanceID", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedGeometryIndex, "RayQuery_CommittedGeometryIndex", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedPrimitiveIndex, "RayQuery_CommittedPrimitiveIndex", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedObjectRayOrigin, "RayQuery_CommittedObjectRayOrigin", OCC::RayQuery_StateVector,     "rayQuery_StateVector",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedObjectRayDirection, "RayQuery_CommittedObjectRayDirection", OCC::RayQuery_StateVector,     "rayQuery_StateVector",      { false, false,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },

  // Raytracing object space uint System Values, raytracing tier 1.1                                                         void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::GeometryIndex,           "GeometryIndex",            OCC::GeometryIndex,            "geometryIndex",             { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadNone, },

  // Inline Ray Query                                                                                                        void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::RayQuery_CandidateInstanceContributionToHitGroupIndex, "RayQuery_CandidateInstanceContributionToHitGroupIndex", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },
  {  OC::RayQuery_CommittedInstanceContributionToHitGroupIndex, "RayQuery_CommittedInstanceContributionToHitGroupIndex", OCC::RayQuery_StateScalar,     "rayQuery_StateScalar",      { false, false, false, false, false, false, false,  true, false, false, false}, Attribute::ReadOnly, },

  // Get handle from heap                                                                                                    void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::AnnotateHandle,          "AnnotateHandle",           OCC::AnnotateHandle,           "annotateHandle",            {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::CreateHandleFromBinding, "CreateHandleFromBinding",  OCC::CreateHandleFromBinding,  "createHandleFromBinding",   {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },
  {  OC::CreateHandleFromHeap,    "CreateHandleFromHeap",     OCC::CreateHandleFromHeap,     "createHandleFromHeap",      {  true, false, false, false, false, false, false, false, false, false, false}, Attribute::ReadNone, },

  // Unpacking intrinsics                                                                                                    void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Unpack4x8,               "Unpack4x8",                OCC::Unpack4x8,                "unpack4x8",                 { false, false, false, false, false, false,  true,  true, false, false, false}, Attribute::ReadNone, },

  // Packing intrinsics                                                                                                      void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::Pack4x8,                 "Pack4x8",                  OCC::Pack4x8,                  "pack4x8",                   { false, false, false, false, false, false,  true,  true, false, false, false}, Attribute::ReadNone, },

  // Helper Lanes                                                                                                            void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::IsHelperLane,            "IsHelperLane",             OCC::IsHelperLane,             "isHelperLane",              { false, false, false, false,  true, false, false, false, false, false, false}, Attribute::ReadOnly, },

  // Quad Wave Ops                                                                                                           void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::QuadVote,                "QuadVote",                 OCC::QuadVote,                 "quadVote",                  { false, false, false, false,  true, false, false, false, false, false, false}, Attribute::None,     },

  // Resources - gather                                                                                                      void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::TextureGatherRaw,        "TextureGatherRaw",         OCC::TextureGatherRaw,         "textureGatherRaw",          { false, false, false, false, false, false,  true,  true,  true, false, false}, Attribute::ReadOnly, },

  // Resources - sample                                                                                                      void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::SampleCmpLevel,          "SampleCmpLevel",           OCC::SampleCmpLevel,           "sampleCmpLevel",            { false,  true,  true, false, false, false, false, false, false, false, false}, Attribute::ReadOnly, },

  // Resources                                                                                                               void,     h,     f,     d,    i1,    i8,   i16,   i32,   i64,   udt,   obj ,  function attribute
  {  OC::TextureStoreSample,      "TextureStoreSample",       OCC::TextureStoreSample,       "textureStoreSample",        { false,  true,  true, false, false, false,  true,  true, false, false, false}, Attribute::None,     },
};
// OPCODE-OLOADS:END

const char *OP::m_OverloadTypeName[kNumTypeOverloads] = {
  "void", "f16", "f32", "f64", "i1", "i8", "i16", "i32", "i64", "udt",
};

const char *OP::m_NamePrefix = "dx.op.";
const char *OP::m_TypePrefix = "dx.types.";
const char *OP::m_MatrixTypePrefix = "class.matrix."; // Allowed in library

// Keep sync with DXIL::AtomicBinOpCode
static const char *AtomicBinOpCodeName[] = {
    "AtomicAdd",
    "AtomicAnd",
    "AtomicOr",
    "AtomicXor",
    "AtomicIMin",
    "AtomicIMax",
    "AtomicUMin",
    "AtomicUMax",
    "AtomicExchange",
    "AtomicInvalid"           // Must be last.
};

unsigned OP::GetTypeSlot(Type *pType) {
  Type::TypeID T = pType->getTypeID();
  switch (T) {
  case Type::VoidTyID:    return 0;
  case Type::HalfTyID:    return 1;
  case Type::FloatTyID:   return 2;
  case Type::DoubleTyID:  return 3;
  case Type::IntegerTyID: {
    IntegerType *pIT = dyn_cast<IntegerType>(pType);
    unsigned Bits = pIT->getBitWidth();
    switch (Bits) {
    case 1:               return 4;
    case 8:               return 5;
    case 16:              return 6;
    case 32:              return 7;
    case 64:              return 8;
    }
  }
  case Type::PointerTyID: return 9;
  case Type::StructTyID:  return 10;
  default:
    break;
  }
  return UINT_MAX;
}

const char *OP::GetOverloadTypeName(unsigned TypeSlot) {
  DXASSERT(TypeSlot < kUserDefineTypeSlot, "otherwise caller passed OOB index");
  return m_OverloadTypeName[TypeSlot];
}

llvm::StringRef OP::GetTypeName(Type *Ty, std::string &str) {
  unsigned TypeSlot = OP::GetTypeSlot(Ty);
  if (TypeSlot < kUserDefineTypeSlot) {
    return GetOverloadTypeName(TypeSlot);
  } else if (TypeSlot == kUserDefineTypeSlot) {
    if (Ty->isPointerTy())
      Ty = Ty->getPointerElementType();
    StructType *ST = cast<StructType>(Ty);
    return ST->getStructName();
  } else if (TypeSlot == kObjectTypeSlot) {
    StructType *ST = cast<StructType>(Ty);
    return ST->getStructName();
  } else {
    raw_string_ostream os(str);
    Ty->print(os);
    os.flush();
    return str;
  }
}

llvm::StringRef OP::ConstructOverloadName(Type *Ty, DXIL::OpCode opCode,
                                          std::string &funcNameStorage) {
  if (Ty == Type::getVoidTy(Ty->getContext())) {
    funcNameStorage = (Twine(OP::m_NamePrefix) + Twine(GetOpCodeClassName(opCode))).str();
  } else {
    funcNameStorage = (Twine(OP::m_NamePrefix) + Twine(GetOpCodeClassName(opCode)) +
                       "." + GetTypeName(Ty, funcNameStorage)).str();
  }
  return funcNameStorage;
}

const char *OP::GetOpCodeName(OpCode opCode) {
  DXASSERT(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes, "otherwise caller passed OOB index");
  return m_OpCodeProps[(unsigned)opCode].pOpCodeName;
}

const char *OP::GetAtomicOpName(DXIL::AtomicBinOpCode OpCode) {
  unsigned opcode = static_cast<unsigned>(OpCode);
  DXASSERT_LOCALVAR(opcode, opcode < static_cast<unsigned>(DXIL::AtomicBinOpCode::Invalid), "otherwise caller passed OOB index");
  return AtomicBinOpCodeName[static_cast<unsigned>(OpCode)];
}

OP::OpCodeClass OP::GetOpCodeClass(OpCode opCode) {
  DXASSERT(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes, "otherwise caller passed OOB index");
  return m_OpCodeProps[(unsigned)opCode].opCodeClass;
}

const char *OP::GetOpCodeClassName(OpCode opCode) {
  DXASSERT(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes, "otherwise caller passed OOB index");
  return m_OpCodeProps[(unsigned)opCode].pOpCodeClassName;
}

llvm::Attribute::AttrKind OP::GetMemAccessAttr(OpCode opCode) {
  DXASSERT(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes, "otherwise caller passed OOB index");
  return m_OpCodeProps[(unsigned)opCode].FuncAttr;
}

bool OP::IsOverloadLegal(OpCode opCode, Type *pType) {
  DXASSERT(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes, "otherwise caller passed OOB index");
  unsigned TypeSlot = GetTypeSlot(pType);
  return TypeSlot != UINT_MAX && m_OpCodeProps[(unsigned)opCode].bAllowOverload[TypeSlot];
}

bool OP::CheckOpCodeTable() {
  for (unsigned i = 0; i < (unsigned)OpCode::NumOpCodes; i++) {
    if ((unsigned)m_OpCodeProps[i].opCode != i)
      return false;
  }

  return true;
}

bool OP::IsDxilOpFuncName(StringRef name) {
  return name.startswith(OP::m_NamePrefix);
}

bool OP::IsDxilOpFunc(const llvm::Function *F) {
  // Test for null to allow IsDxilOpFunc(Call.getCalledFunc()) to be resilient to indirect calls
  if (F == nullptr || !F->hasName())
    return false;
  return IsDxilOpFuncName(F->getName());
}

bool OP::IsDxilOpTypeName(StringRef name) {
  return name.startswith(m_TypePrefix) || name.startswith(m_MatrixTypePrefix);
}

bool OP::IsDxilOpType(llvm::StructType *ST) {
  if (!ST->hasName())
    return false;
  StringRef Name = ST->getName();
  return IsDxilOpTypeName(Name);
}

bool OP::IsDupDxilOpType(llvm::StructType *ST) {
  if (!ST->hasName())
    return false;
  StringRef Name = ST->getName();
  if (!IsDxilOpTypeName(Name))
    return false;
  size_t DotPos = Name.rfind('.');
  if (DotPos == 0 || DotPos == StringRef::npos || Name.back() == '.' ||
      !isdigit(static_cast<unsigned char>(Name[DotPos + 1])))
    return false;
  return true;
}

StructType *OP::GetOriginalDxilOpType(llvm::StructType *ST, llvm::Module &M) {
  DXASSERT(IsDupDxilOpType(ST), "else should not call GetOriginalDxilOpType");
  StringRef Name = ST->getName();
  size_t DotPos = Name.rfind('.');
  StructType *OriginalST = M.getTypeByName(Name.substr(0, DotPos));
  DXASSERT(OriginalST, "else name collison without original type");
  DXASSERT(ST->isLayoutIdentical(OriginalST),
           "else invalid layout for dxil types");
  return OriginalST;
}

bool OP::IsDxilOpFuncCallInst(const llvm::Instruction *I) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  if (CI == nullptr) return false;
  return IsDxilOpFunc(CI->getCalledFunction());
}

bool OP::IsDxilOpFuncCallInst(const llvm::Instruction *I, OpCode opcode) {
  if (!IsDxilOpFuncCallInst(I)) return false;
  return llvm::cast<llvm::ConstantInt>(I->getOperand(0))->getZExtValue() == (unsigned)opcode;
}

OP::OpCode OP::GetDxilOpFuncCallInst(const llvm::Instruction *I) {
  DXASSERT(IsDxilOpFuncCallInst(I), "else caller didn't call IsDxilOpFuncCallInst to check");
  return (OP::OpCode)llvm::cast<llvm::ConstantInt>(I->getOperand(0))->getZExtValue();
}

bool OP::IsDxilOpWave(OpCode C) {
  unsigned op = (unsigned)C;
  /* <py::lines('OPCODE-WAVE')>hctdb_instrhelp.get_instrs_pred("op", "is_wave")</py>*/
  // OPCODE-WAVE:BEGIN
  // Instructions: WaveIsFirstLane=110, WaveGetLaneIndex=111,
  // WaveGetLaneCount=112, WaveAnyTrue=113, WaveAllTrue=114,
  // WaveActiveAllEqual=115, WaveActiveBallot=116, WaveReadLaneAt=117,
  // WaveReadLaneFirst=118, WaveActiveOp=119, WaveActiveBit=120,
  // WavePrefixOp=121, QuadReadLaneAt=122, QuadOp=123, WaveAllBitCount=135,
  // WavePrefixBitCount=136, WaveMatch=165, WaveMultiPrefixOp=166,
  // WaveMultiPrefixBitCount=167, QuadVote=222
  return (110 <= op && op <= 123) || (135 <= op && op <= 136) || (165 <= op && op <= 167) || op == 222;
  // OPCODE-WAVE:END
}

bool OP::IsDxilOpGradient(OpCode C) {
  unsigned op = (unsigned)C;
  /* <py::lines('OPCODE-GRADIENT')>hctdb_instrhelp.get_instrs_pred("op", "is_gradient")</py>*/
  // OPCODE-GRADIENT:BEGIN
  // Instructions: Sample=60, SampleBias=61, SampleCmp=64, CalculateLOD=81,
  // DerivCoarseX=83, DerivCoarseY=84, DerivFineX=85, DerivFineY=86,
  // WriteSamplerFeedback=174, WriteSamplerFeedbackBias=175
  return (60 <= op && op <= 61) || op == 64 || op == 81 || (83 <= op && op <= 86) || (174 <= op && op <= 175);
  // OPCODE-GRADIENT:END
}

bool OP::IsDxilOpFeedback(OpCode C) {
  unsigned op = (unsigned)C;
  /* <py::lines('OPCODE-FEEDBACK')>hctdb_instrhelp.get_instrs_pred("op", "is_feedback")</py>*/
  // OPCODE-FEEDBACK:BEGIN
  // Instructions: WriteSamplerFeedback=174, WriteSamplerFeedbackBias=175,
  // WriteSamplerFeedbackLevel=176, WriteSamplerFeedbackGrad=177
  return (174 <= op && op <= 177);
  // OPCODE-FEEDBACK:END
}

#define SFLAG(stage) ((unsigned)1 << (unsigned)DXIL::ShaderKind::stage)
void OP::GetMinShaderModelAndMask(OpCode C, bool bWithTranslation,
                                  unsigned &major, unsigned &minor,
                                  unsigned &mask) {
  unsigned op = (unsigned)C;
  // Default is 6.0, all stages
  major = 6;  minor = 0;
  mask = ((unsigned)1 << (unsigned)DXIL::ShaderKind::Invalid) - 1;
  /* <py::lines('OPCODE-SMMASK')>hctdb_instrhelp.get_min_sm_and_mask_text()</py>*/
  // OPCODE-SMMASK:BEGIN
  // Instructions: ThreadId=93, GroupId=94, ThreadIdInGroup=95,
  // FlattenedThreadIdInGroup=96
  if ((93 <= op && op <= 96)) {
    mask = SFLAG(Compute) | SFLAG(Mesh) | SFLAG(Amplification);
    return;
  }
  // Instructions: DomainLocation=105
  if (op == 105) {
    mask = SFLAG(Domain);
    return;
  }
  // Instructions: LoadOutputControlPoint=103, LoadPatchConstant=104
  if ((103 <= op && op <= 104)) {
    mask = SFLAG(Domain) | SFLAG(Hull);
    return;
  }
  // Instructions: EmitStream=97, CutStream=98, EmitThenCutStream=99,
  // GSInstanceID=100
  if ((97 <= op && op <= 100)) {
    mask = SFLAG(Geometry);
    return;
  }
  // Instructions: PrimitiveID=108
  if (op == 108) {
    mask = SFLAG(Geometry) | SFLAG(Domain) | SFLAG(Hull);
    return;
  }
  // Instructions: StorePatchConstant=106, OutputControlPointID=107
  if ((106 <= op && op <= 107)) {
    mask = SFLAG(Hull);
    return;
  }
  // Instructions: QuadReadLaneAt=122, QuadOp=123
  if ((122 <= op && op <= 123)) {
    mask = SFLAG(Library) | SFLAG(Compute) | SFLAG(Amplification) | SFLAG(Mesh) | SFLAG(Pixel);
    return;
  }
  // Instructions: WaveIsFirstLane=110, WaveGetLaneIndex=111,
  // WaveGetLaneCount=112, WaveAnyTrue=113, WaveAllTrue=114,
  // WaveActiveAllEqual=115, WaveActiveBallot=116, WaveReadLaneAt=117,
  // WaveReadLaneFirst=118, WaveActiveOp=119, WaveActiveBit=120,
  // WavePrefixOp=121, WaveAllBitCount=135, WavePrefixBitCount=136
  if ((110 <= op && op <= 121) || (135 <= op && op <= 136)) {
    mask = SFLAG(Library) | SFLAG(Compute) | SFLAG(Amplification) | SFLAG(Mesh) | SFLAG(Pixel) | SFLAG(Vertex) | SFLAG(Hull) | SFLAG(Domain) | SFLAG(Geometry) | SFLAG(RayGeneration) | SFLAG(Intersection) | SFLAG(AnyHit) | SFLAG(ClosestHit) | SFLAG(Miss) | SFLAG(Callable);
    return;
  }
  // Instructions: Sample=60, SampleBias=61, SampleCmp=64, CalculateLOD=81,
  // DerivCoarseX=83, DerivCoarseY=84, DerivFineX=85, DerivFineY=86
  if ((60 <= op && op <= 61) || op == 64 || op == 81 || (83 <= op && op <= 86)) {
    mask = SFLAG(Library) | SFLAG(Pixel) | SFLAG(Compute) | SFLAG(Amplification) | SFLAG(Mesh);
    return;
  }
  // Instructions: RenderTargetGetSamplePosition=76,
  // RenderTargetGetSampleCount=77, Discard=82, EvalSnapped=87,
  // EvalSampleIndex=88, EvalCentroid=89, SampleIndex=90, Coverage=91,
  // InnerCoverage=92
  if ((76 <= op && op <= 77) || op == 82 || (87 <= op && op <= 92)) {
    mask = SFLAG(Pixel);
    return;
  }
  // Instructions: AttributeAtVertex=137
  if (op == 137) {
    major = 6;  minor = 1;
    mask = SFLAG(Pixel);
    return;
  }
  // Instructions: ViewID=138
  if (op == 138) {
    major = 6;  minor = 1;
    mask = SFLAG(Vertex) | SFLAG(Hull) | SFLAG(Domain) | SFLAG(Geometry) | SFLAG(Pixel) | SFLAG(Mesh);
    return;
  }
  // Instructions: RawBufferLoad=139, RawBufferStore=140
  if ((139 <= op && op <= 140)) {
    if (bWithTranslation) {
      major = 6;  minor = 0;
    } else {
      major = 6;  minor = 2;
    }
    return;
  }
  // Instructions: IgnoreHit=155, AcceptHitAndEndSearch=156
  if ((155 <= op && op <= 156)) {
    major = 6;  minor = 3;
    mask = SFLAG(AnyHit);
    return;
  }
  // Instructions: CallShader=159
  if (op == 159) {
    major = 6;  minor = 3;
    mask = SFLAG(Library) | SFLAG(ClosestHit) | SFLAG(RayGeneration) | SFLAG(Miss) | SFLAG(Callable);
    return;
  }
  // Instructions: ReportHit=158
  if (op == 158) {
    major = 6;  minor = 3;
    mask = SFLAG(Library) | SFLAG(Intersection);
    return;
  }
  // Instructions: InstanceID=141, InstanceIndex=142, HitKind=143,
  // ObjectRayOrigin=149, ObjectRayDirection=150, ObjectToWorld=151,
  // WorldToObject=152, PrimitiveIndex=161
  if ((141 <= op && op <= 143) || (149 <= op && op <= 152) || op == 161) {
    major = 6;  minor = 3;
    mask = SFLAG(Library) | SFLAG(Intersection) | SFLAG(AnyHit) | SFLAG(ClosestHit);
    return;
  }
  // Instructions: RayFlags=144, WorldRayOrigin=147, WorldRayDirection=148,
  // RayTMin=153, RayTCurrent=154
  if (op == 144 || (147 <= op && op <= 148) || (153 <= op && op <= 154)) {
    major = 6;  minor = 3;
    mask = SFLAG(Library) | SFLAG(Intersection) | SFLAG(AnyHit) | SFLAG(ClosestHit) | SFLAG(Miss);
    return;
  }
  // Instructions: TraceRay=157
  if (op == 157) {
    major = 6;  minor = 3;
    mask = SFLAG(Library) | SFLAG(RayGeneration) | SFLAG(ClosestHit) | SFLAG(Miss);
    return;
  }
  // Instructions: DispatchRaysIndex=145, DispatchRaysDimensions=146
  if ((145 <= op && op <= 146)) {
    major = 6;  minor = 3;
    mask = SFLAG(Library) | SFLAG(RayGeneration) | SFLAG(Intersection) | SFLAG(AnyHit) | SFLAG(ClosestHit) | SFLAG(Miss) | SFLAG(Callable);
    return;
  }
  // Instructions: CreateHandleForLib=160
  if (op == 160) {
    if (bWithTranslation) {
      major = 6;  minor = 0;
    } else {
      major = 6;  minor = 3;
    }
    return;
  }
  // Instructions: Dot2AddHalf=162, Dot4AddI8Packed=163, Dot4AddU8Packed=164
  if ((162 <= op && op <= 164)) {
    major = 6;  minor = 4;
    return;
  }
  // Instructions: WriteSamplerFeedbackLevel=176, WriteSamplerFeedbackGrad=177,
  // AllocateRayQuery=178, RayQuery_TraceRayInline=179, RayQuery_Proceed=180,
  // RayQuery_Abort=181, RayQuery_CommitNonOpaqueTriangleHit=182,
  // RayQuery_CommitProceduralPrimitiveHit=183, RayQuery_CommittedStatus=184,
  // RayQuery_CandidateType=185, RayQuery_CandidateObjectToWorld3x4=186,
  // RayQuery_CandidateWorldToObject3x4=187,
  // RayQuery_CommittedObjectToWorld3x4=188,
  // RayQuery_CommittedWorldToObject3x4=189,
  // RayQuery_CandidateProceduralPrimitiveNonOpaque=190,
  // RayQuery_CandidateTriangleFrontFace=191,
  // RayQuery_CommittedTriangleFrontFace=192,
  // RayQuery_CandidateTriangleBarycentrics=193,
  // RayQuery_CommittedTriangleBarycentrics=194, RayQuery_RayFlags=195,
  // RayQuery_WorldRayOrigin=196, RayQuery_WorldRayDirection=197,
  // RayQuery_RayTMin=198, RayQuery_CandidateTriangleRayT=199,
  // RayQuery_CommittedRayT=200, RayQuery_CandidateInstanceIndex=201,
  // RayQuery_CandidateInstanceID=202, RayQuery_CandidateGeometryIndex=203,
  // RayQuery_CandidatePrimitiveIndex=204, RayQuery_CandidateObjectRayOrigin=205,
  // RayQuery_CandidateObjectRayDirection=206,
  // RayQuery_CommittedInstanceIndex=207, RayQuery_CommittedInstanceID=208,
  // RayQuery_CommittedGeometryIndex=209, RayQuery_CommittedPrimitiveIndex=210,
  // RayQuery_CommittedObjectRayOrigin=211,
  // RayQuery_CommittedObjectRayDirection=212,
  // RayQuery_CandidateInstanceContributionToHitGroupIndex=214,
  // RayQuery_CommittedInstanceContributionToHitGroupIndex=215
  if ((176 <= op && op <= 212) || (214 <= op && op <= 215)) {
    major = 6;  minor = 5;
    return;
  }
  // Instructions: DispatchMesh=173
  if (op == 173) {
    major = 6;  minor = 5;
    mask = SFLAG(Amplification);
    return;
  }
  // Instructions: WaveMatch=165, WaveMultiPrefixOp=166,
  // WaveMultiPrefixBitCount=167
  if ((165 <= op && op <= 167)) {
    major = 6;  minor = 5;
    mask = SFLAG(Library) | SFLAG(Compute) | SFLAG(Amplification) | SFLAG(Mesh) | SFLAG(Pixel) | SFLAG(Vertex) | SFLAG(Hull) | SFLAG(Domain) | SFLAG(Geometry) | SFLAG(RayGeneration) | SFLAG(Intersection) | SFLAG(AnyHit) | SFLAG(ClosestHit) | SFLAG(Miss) | SFLAG(Callable);
    return;
  }
  // Instructions: GeometryIndex=213
  if (op == 213) {
    major = 6;  minor = 5;
    mask = SFLAG(Library) | SFLAG(Intersection) | SFLAG(AnyHit) | SFLAG(ClosestHit);
    return;
  }
  // Instructions: WriteSamplerFeedback=174, WriteSamplerFeedbackBias=175
  if ((174 <= op && op <= 175)) {
    major = 6;  minor = 5;
    mask = SFLAG(Library) | SFLAG(Pixel);
    return;
  }
  // Instructions: SetMeshOutputCounts=168, EmitIndices=169, GetMeshPayload=170,
  // StoreVertexOutput=171, StorePrimitiveOutput=172
  if ((168 <= op && op <= 172)) {
    major = 6;  minor = 5;
    mask = SFLAG(Mesh);
    return;
  }
  // Instructions: CreateHandleFromHeap=218, Unpack4x8=219, Pack4x8=220,
  // IsHelperLane=221
  if ((218 <= op && op <= 221)) {
    major = 6;  minor = 6;
    return;
  }
  // Instructions: AnnotateHandle=216, CreateHandleFromBinding=217
  if ((216 <= op && op <= 217)) {
    if (bWithTranslation) {
      major = 6;  minor = 0;
    } else {
      major = 6;  minor = 6;
    }
    return;
  }
  // Instructions: TextureGatherRaw=223, SampleCmpLevel=224,
  // TextureStoreSample=225
  if ((223 <= op && op <= 225)) {
    major = 6;  minor = 7;
    return;
  }
  // Instructions: QuadVote=222
  if (op == 222) {
    if (bWithTranslation) {
      major = 6;  minor = 0;
    } else {
      major = 6;  minor = 7;
    }
    mask = SFLAG(Library) | SFLAG(Compute) | SFLAG(Amplification) | SFLAG(Mesh) | SFLAG(Pixel);
    return;
  }
  // OPCODE-SMMASK:END
}

void OP::GetMinShaderModelAndMask(const llvm::CallInst *CI, bool bWithTranslation,
                                  unsigned valMajor, unsigned valMinor,
                                  unsigned &major, unsigned &minor,
                                  unsigned &mask) {
  OpCode opcode = OP::GetDxilOpFuncCallInst(CI);
  GetMinShaderModelAndMask(opcode, bWithTranslation, major, minor, mask);

  unsigned op = (unsigned)opcode;
  // These ops cannot indicate support for CS, AS, or MS,
  // otherwise, it's saying these are guaranteed to be supported
  // on the lowest shader model returned by this function
  // for these shader stages.  For CS, SM 6.6 is required,
  // and for AS/MS, an optional feature is required.
  // This also breaks compatibility for existing validators.
  // We need a different mechanism to be supported in functions
  // for runtime linking.
  // Instructions: Sample=60, SampleBias=61, SampleCmp=64, CalculateLOD=81,
  // DerivCoarseX=83, DerivCoarseY=84, DerivFineX=85, DerivFineY=86
  if ((60 <= op && op <= 61) || op == 64 || op == 81 || (83 <= op && op <= 86)) {
    mask &= ~(SFLAG(Compute) | SFLAG(Amplification) | SFLAG(Mesh));
    return;
  }

  if (DXIL::CompareVersions(valMajor, valMinor, 1, 5) < 0) {
    // validator 1.4 didn't exclude wave ops in mask
    if (IsDxilOpWave(opcode))
      mask = ((unsigned)1 << (unsigned)DXIL::ShaderKind::Invalid) - 1;
    // These shader models don't exist before 1.5
    mask &= ~(SFLAG(Amplification) | SFLAG(Mesh));
    // validator 1.4 didn't have any additional rules applied:
    return;
  }

  // Additional rules are applied manually here.

  // Barrier with mode != UAVFenceGlobal requires compute, amplification, or mesh
  // Instructions: Barrier=80
  if (opcode == DXIL::OpCode::Barrier) {
    DxilInst_Barrier barrier(const_cast<CallInst*>(CI));
    unsigned mode = barrier.get_barrierMode_val();
    if (mode != (unsigned)DXIL::BarrierMode::UAVFenceGlobal) {
      mask = SFLAG(Library) | SFLAG(Compute) | SFLAG(Amplification) | SFLAG(Mesh);
    }
    return;
  }

  // 64-bit integer atomic ops require 6.6
  else if (opcode == DXIL::OpCode::AtomicBinOp || opcode == DXIL::OpCode::AtomicCompareExchange) {
    Type *pOverloadType = GetOverloadType(opcode, CI->getCalledFunction());
    if (pOverloadType->isIntegerTy(64)) {
      major = 6;
      minor = 6;
    }
  }

  // AnnotateHandle and CreateHandleFromBinding can be translated down to
  // SM 6.0, but this wasn't set properly in validator version 6.6, so make it
  // match when using that version.
  else if (bWithTranslation &&
           DXIL::CompareVersions(valMajor, valMinor, 1, 6) == 0 &&
           (opcode == DXIL::OpCode::AnnotateHandle ||
            opcode == DXIL::OpCode::CreateHandleFromBinding)) {
    major = 6;
    minor = 6;
  }
}
#undef SFLAG


static Type *GetOrCreateStructType(LLVMContext &Ctx, ArrayRef<Type*> types, StringRef Name, Module *pModule) {
  if (StructType *ST = pModule->getTypeByName(Name)) {
    // TODO: validate the exist type match types if needed.
    return ST;
  }
  else
    return StructType::create(Ctx, types, Name);
}

//------------------------------------------------------------------------------
//
//  OP methods.
//
OP::OP(LLVMContext &Ctx, Module *pModule)
: m_Ctx(Ctx)
, m_pModule(pModule)
, m_LowPrecisionMode(DXIL::LowPrecisionMode::Undefined) {
  memset(m_pResRetType, 0, sizeof(m_pResRetType));
  memset(m_pCBufferRetType, 0, sizeof(m_pCBufferRetType));
  memset(m_OpCodeClassCache, 0, sizeof(m_OpCodeClassCache));
  static_assert(_countof(OP::m_OpCodeProps) == (size_t)OP::OpCode::NumOpCodes, "forgot to update OP::m_OpCodeProps");

  m_pHandleType = GetOrCreateStructType(m_Ctx, Type::getInt8PtrTy(m_Ctx),
                                        "dx.types.Handle", pModule);
  m_pResourcePropertiesType = GetOrCreateStructType(
      m_Ctx, {Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx)},
      "dx.types.ResourceProperties", pModule);

  m_pResourceBindingType =
      GetOrCreateStructType(m_Ctx,
                            {Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx),
                             Type::getInt32Ty(m_Ctx), Type::getInt8Ty(m_Ctx)},
                            "dx.types.ResBind", pModule);

  Type *DimsType[4] = { Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx) };
  m_pDimensionsType = GetOrCreateStructType(m_Ctx, DimsType, "dx.types.Dimensions", pModule);

  Type *SamplePosType[2] = { Type::getFloatTy(m_Ctx), Type::getFloatTy(m_Ctx) };
  m_pSamplePosType = GetOrCreateStructType(m_Ctx, SamplePosType, "dx.types.SamplePos", pModule);

  Type *I32cTypes[2] = { Type::getInt32Ty(m_Ctx), Type::getInt1Ty(m_Ctx) };
  m_pBinaryWithCarryType = GetOrCreateStructType(m_Ctx, I32cTypes, "dx.types.i32c", pModule);

  Type *TwoI32Types[2] = { Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx) };
  m_pBinaryWithTwoOutputsType = GetOrCreateStructType(m_Ctx, TwoI32Types, "dx.types.twoi32", pModule);

  Type *SplitDoubleTypes[2] = { Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx) }; // Lo, Hi.
  m_pSplitDoubleType = GetOrCreateStructType(m_Ctx, SplitDoubleTypes, "dx.types.splitdouble", pModule);

  Type *FourI32Types[4] = { Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx), Type::getInt32Ty(m_Ctx) }; // HiHi, HiLo, LoHi, LoLo
  m_pFourI32Type = GetOrCreateStructType(m_Ctx, FourI32Types, "dx.types.fouri32", pModule);

  Type *FourI16Types[4] = { Type::getInt16Ty(m_Ctx), Type::getInt16Ty(m_Ctx), Type::getInt16Ty(m_Ctx), Type::getInt16Ty(m_Ctx) }; // HiHi, HiLo, LoHi, LoLo
  m_pFourI16Type = GetOrCreateStructType(m_Ctx, FourI16Types, "dx.types.fouri16", pModule);

  // When loading a module into an existing context where types are merged,
  // type names may change.  When this happens, any intrinsics overloaded on
  // UDT types will no longer have matching overload names.
  // This causes RefreshCache() to assert.
  // This fixes the function names to they match the expected types,
  // preventing RefreshCache() from failing due to this issue.
  FixOverloadNames();

  // Try to find existing intrinsic function.
  RefreshCache();
}

void OP::RefreshCache() {
  for (Function &F : m_pModule->functions()) {
    if (OP::IsDxilOpFunc(&F) && !F.user_empty()) {
      CallInst *CI = cast<CallInst>(*F.user_begin());
      OpCode OpCode = OP::GetDxilOpFuncCallInst(CI);
      Type *pOverloadType = OP::GetOverloadType(OpCode, &F);
      Function *OpFunc = GetOpFunc(OpCode, pOverloadType);
      (void)(OpFunc);
      DXASSERT_NOMSG(OpFunc == &F);
    }
  }
}

void OP::FixOverloadNames() {
  // When merging code from multiple sources, such as with linking,
  // type names that collide, but don't have the same type will be
  // automically renamed with .0+ name disambiguation.  However,
  // DXIL intrinsic overloads will not be renamed to disambiguate them,
  // since they exist in separate modules at the time.
  // This leads to name collisions between different types when linking.
  // Do this after loading into a shared context, and before copying
  // code into a common module, to prevent this problem.
  for (Function &F : m_pModule->functions()) {
    if (F.isDeclaration() && OP::IsDxilOpFunc(&F) && !F.user_empty()) {
      CallInst *CI = cast<CallInst>(*F.user_begin());
      DXIL::OpCode opCode = OP::GetDxilOpFuncCallInst(CI);
      llvm::Type *Ty = OP::GetOverloadType(opCode, &F);
      if (isa<StructType>(Ty) || isa<PointerType>(Ty)) {
        std::string funcName;
        if (OP::ConstructOverloadName(Ty, opCode, funcName).compare(F.getName()) != 0) {
          F.setName(funcName);
        }
      }
    }
  }
}

void OP::UpdateCache(OpCodeClass opClass, Type * Ty, llvm::Function *F) {
  m_OpCodeClassCache[(unsigned)opClass].pOverloads[Ty] = F;
  m_FunctionToOpClass[F] = opClass;
}

Function *OP::GetOpFunc(OpCode opCode, Type *pOverloadType) {
  DXASSERT(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes, "otherwise caller passed OOB OpCode");
  _Analysis_assume_(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes);
  DXASSERT(IsOverloadLegal(opCode, pOverloadType), "otherwise the caller requested illegal operation overload (eg HLSL function with unsupported types for mapped intrinsic function)");
  OpCodeClass opClass = m_OpCodeProps[(unsigned)opCode].opCodeClass;
  Function *&F = m_OpCodeClassCache[(unsigned)opClass].pOverloads[pOverloadType];
  if (F != nullptr) {
    UpdateCache(opClass, pOverloadType, F);
    return F;
  }

  vector<Type*> ArgTypes;      // RetType is ArgTypes[0]
  Type *pETy = pOverloadType;
  Type *pRes = GetHandleType();
  Type *pDim = GetDimensionsType();
  Type *pPos = GetSamplePosType();
  Type *pV = Type::getVoidTy(m_Ctx);
  Type *pI1 = Type::getInt1Ty(m_Ctx);
  Type *pI8 = Type::getInt8Ty(m_Ctx);
  Type *pI16 = Type::getInt16Ty(m_Ctx);
  Type *pI32 = Type::getInt32Ty(m_Ctx);
  Type *pPI32 = Type::getInt32PtrTy(m_Ctx); (void)(pPI32); // Currently unused.
  Type *pI64 = Type::getInt64Ty(m_Ctx); (void)(pI64); // Currently unused.
  Type *pF16 = Type::getHalfTy(m_Ctx);
  Type *pF32 = Type::getFloatTy(m_Ctx);
  Type *pPF32 = Type::getFloatPtrTy(m_Ctx);
  Type *pI32C = GetBinaryWithCarryType();
  Type *p2I32 = GetBinaryWithTwoOutputsType();
  Type *pF64 = Type::getDoubleTy(m_Ctx);
  Type *pSDT = GetSplitDoubleType();  // Split double type.
  Type *p4I32 = GetFourI32Type(); // 4 i32s in a struct.
  
  Type *udt = pOverloadType;
  Type *obj = pOverloadType;
  Type *resProperty = GetResourcePropertiesType();
  Type *resBind = GetResourceBindingType();

  std::string funcName;
  ConstructOverloadName(pOverloadType, opCode, funcName);

  // Try to find exist function with the same name in the module.
  if (Function *existF = m_pModule->getFunction(funcName)) {
    F = existF;
    UpdateCache(opClass, pOverloadType, F);
    return F;
  }

#define A(_x) ArgTypes.emplace_back(_x)
#define RRT(_y) A(GetResRetType(_y))
#define CBRT(_y) A(GetCBufferRetType(_y))
#define VEC4(_y) A(GetVectorType(4,_y))

/* <py::lines('OPCODE-OLOAD-FUNCS')>hctdb_instrhelp.get_oloads_funcs()</py>*/
  switch (opCode) {            // return     opCode
// OPCODE-OLOAD-FUNCS:BEGIN
    // Temporary, indexable, input, output registers
  case OpCode::TempRegLoad:            A(pETy);     A(pI32); A(pI32); break;
  case OpCode::TempRegStore:           A(pV);       A(pI32); A(pI32); A(pETy); break;
  case OpCode::MinPrecXRegLoad:        A(pETy);     A(pI32); A(pPF32);A(pI32); A(pI8);  break;
  case OpCode::MinPrecXRegStore:       A(pV);       A(pI32); A(pPF32);A(pI32); A(pI8);  A(pETy); break;
  case OpCode::LoadInput:              A(pETy);     A(pI32); A(pI32); A(pI32); A(pI8);  A(pI32); break;
  case OpCode::StoreOutput:            A(pV);       A(pI32); A(pI32); A(pI32); A(pI8);  A(pETy); break;

    // Unary float
  case OpCode::FAbs:                   A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Saturate:               A(pETy);     A(pI32); A(pETy); break;
  case OpCode::IsNaN:                  A(pI1);      A(pI32); A(pETy); break;
  case OpCode::IsInf:                  A(pI1);      A(pI32); A(pETy); break;
  case OpCode::IsFinite:               A(pI1);      A(pI32); A(pETy); break;
  case OpCode::IsNormal:               A(pI1);      A(pI32); A(pETy); break;
  case OpCode::Cos:                    A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Sin:                    A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Tan:                    A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Acos:                   A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Asin:                   A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Atan:                   A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Hcos:                   A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Hsin:                   A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Htan:                   A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Exp:                    A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Frc:                    A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Log:                    A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Sqrt:                   A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Rsqrt:                  A(pETy);     A(pI32); A(pETy); break;

    // Unary float - rounding
  case OpCode::Round_ne:               A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Round_ni:               A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Round_pi:               A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Round_z:                A(pETy);     A(pI32); A(pETy); break;

    // Unary int
  case OpCode::Bfrev:                  A(pETy);     A(pI32); A(pETy); break;
  case OpCode::Countbits:              A(pI32);     A(pI32); A(pETy); break;
  case OpCode::FirstbitLo:             A(pI32);     A(pI32); A(pETy); break;

    // Unary uint
  case OpCode::FirstbitHi:             A(pI32);     A(pI32); A(pETy); break;

    // Unary int
  case OpCode::FirstbitSHi:            A(pI32);     A(pI32); A(pETy); break;

    // Binary float
  case OpCode::FMax:                   A(pETy);     A(pI32); A(pETy); A(pETy); break;
  case OpCode::FMin:                   A(pETy);     A(pI32); A(pETy); A(pETy); break;

    // Binary int
  case OpCode::IMax:                   A(pETy);     A(pI32); A(pETy); A(pETy); break;
  case OpCode::IMin:                   A(pETy);     A(pI32); A(pETy); A(pETy); break;

    // Binary uint
  case OpCode::UMax:                   A(pETy);     A(pI32); A(pETy); A(pETy); break;
  case OpCode::UMin:                   A(pETy);     A(pI32); A(pETy); A(pETy); break;

    // Binary int with two outputs
  case OpCode::IMul:                   A(p2I32);    A(pI32); A(pETy); A(pETy); break;

    // Binary uint with two outputs
  case OpCode::UMul:                   A(p2I32);    A(pI32); A(pETy); A(pETy); break;
  case OpCode::UDiv:                   A(p2I32);    A(pI32); A(pETy); A(pETy); break;

    // Binary uint with carry or borrow
  case OpCode::UAddc:                  A(pI32C);    A(pI32); A(pETy); A(pETy); break;
  case OpCode::USubb:                  A(pI32C);    A(pI32); A(pETy); A(pETy); break;

    // Tertiary float
  case OpCode::FMad:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); break;
  case OpCode::Fma:                    A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); break;

    // Tertiary int
  case OpCode::IMad:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); break;

    // Tertiary uint
  case OpCode::UMad:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); break;

    // Tertiary int
  case OpCode::Msad:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); break;
  case OpCode::Ibfe:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); break;

    // Tertiary uint
  case OpCode::Ubfe:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); break;

    // Quaternary
  case OpCode::Bfi:                    A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); A(pETy); break;

    // Dot
  case OpCode::Dot2:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); A(pETy); break;
  case OpCode::Dot3:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); A(pETy); A(pETy); A(pETy); break;
  case OpCode::Dot4:                   A(pETy);     A(pI32); A(pETy); A(pETy); A(pETy); A(pETy); A(pETy); A(pETy); A(pETy); A(pETy); break;

    // Resources
  case OpCode::CreateHandle:           A(pRes);     A(pI32); A(pI8);  A(pI32); A(pI32); A(pI1);  break;
  case OpCode::CBufferLoad:            A(pETy);     A(pI32); A(pRes); A(pI32); A(pI32); break;
  case OpCode::CBufferLoadLegacy:      CBRT(pETy);  A(pI32); A(pRes); A(pI32); break;

    // Resources - sample
  case OpCode::Sample:                 RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); A(pF32); break;
  case OpCode::SampleBias:             RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); A(pF32); A(pF32); break;
  case OpCode::SampleLevel:            RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); A(pF32); break;
  case OpCode::SampleGrad:             RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); break;
  case OpCode::SampleCmp:              RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); A(pF32); A(pF32); break;
  case OpCode::SampleCmpLevelZero:     RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); A(pF32); break;

    // Resources
  case OpCode::TextureLoad:            RRT(pETy);   A(pI32); A(pRes); A(pI32); A(pI32); A(pI32); A(pI32); A(pI32); A(pI32); A(pI32); break;
  case OpCode::TextureStore:           A(pV);       A(pI32); A(pRes); A(pI32); A(pI32); A(pI32); A(pETy); A(pETy); A(pETy); A(pETy); A(pI8);  break;
  case OpCode::BufferLoad:             RRT(pETy);   A(pI32); A(pRes); A(pI32); A(pI32); break;
  case OpCode::BufferStore:            A(pV);       A(pI32); A(pRes); A(pI32); A(pI32); A(pETy); A(pETy); A(pETy); A(pETy); A(pI8);  break;
  case OpCode::BufferUpdateCounter:    A(pI32);     A(pI32); A(pRes); A(pI8);  break;
  case OpCode::CheckAccessFullyMapped: A(pI1);      A(pI32); A(pI32); break;
  case OpCode::GetDimensions:          A(pDim);     A(pI32); A(pRes); A(pI32); break;

    // Resources - gather
  case OpCode::TextureGather:          RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); break;
  case OpCode::TextureGatherCmp:       RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); A(pF32); break;

    // Resources - sample
  case OpCode::Texture2DMSGetSamplePosition:A(pPos);     A(pI32); A(pRes); A(pI32); break;
  case OpCode::RenderTargetGetSamplePosition:A(pPos);     A(pI32); A(pI32); break;
  case OpCode::RenderTargetGetSampleCount:A(pI32);     A(pI32); break;

    // Synchronization
  case OpCode::AtomicBinOp:            A(pETy);     A(pI32); A(pRes); A(pI32); A(pI32); A(pI32); A(pI32); A(pETy); break;
  case OpCode::AtomicCompareExchange:  A(pETy);     A(pI32); A(pRes); A(pI32); A(pI32); A(pI32); A(pETy); A(pETy); break;
  case OpCode::Barrier:                A(pV);       A(pI32); A(pI32); break;

    // Derivatives
  case OpCode::CalculateLOD:           A(pF32);     A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pI1);  break;

    // Pixel shader
  case OpCode::Discard:                A(pV);       A(pI32); A(pI1);  break;

    // Derivatives
  case OpCode::DerivCoarseX:           A(pETy);     A(pI32); A(pETy); break;
  case OpCode::DerivCoarseY:           A(pETy);     A(pI32); A(pETy); break;
  case OpCode::DerivFineX:             A(pETy);     A(pI32); A(pETy); break;
  case OpCode::DerivFineY:             A(pETy);     A(pI32); A(pETy); break;

    // Pixel shader
  case OpCode::EvalSnapped:            A(pETy);     A(pI32); A(pI32); A(pI32); A(pI8);  A(pI32); A(pI32); break;
  case OpCode::EvalSampleIndex:        A(pETy);     A(pI32); A(pI32); A(pI32); A(pI8);  A(pI32); break;
  case OpCode::EvalCentroid:           A(pETy);     A(pI32); A(pI32); A(pI32); A(pI8);  break;
  case OpCode::SampleIndex:            A(pI32);     A(pI32); break;
  case OpCode::Coverage:               A(pI32);     A(pI32); break;
  case OpCode::InnerCoverage:          A(pI32);     A(pI32); break;

    // Compute/Mesh/Amplification shader
  case OpCode::ThreadId:               A(pI32);     A(pI32); A(pI32); break;
  case OpCode::GroupId:                A(pI32);     A(pI32); A(pI32); break;
  case OpCode::ThreadIdInGroup:        A(pI32);     A(pI32); A(pI32); break;
  case OpCode::FlattenedThreadIdInGroup:A(pI32);     A(pI32); break;

    // Geometry shader
  case OpCode::EmitStream:             A(pV);       A(pI32); A(pI8);  break;
  case OpCode::CutStream:              A(pV);       A(pI32); A(pI8);  break;
  case OpCode::EmitThenCutStream:      A(pV);       A(pI32); A(pI8);  break;
  case OpCode::GSInstanceID:           A(pI32);     A(pI32); break;

    // Double precision
  case OpCode::MakeDouble:             A(pF64);     A(pI32); A(pI32); A(pI32); break;
  case OpCode::SplitDouble:            A(pSDT);     A(pI32); A(pF64); break;

    // Domain and hull shader
  case OpCode::LoadOutputControlPoint: A(pETy);     A(pI32); A(pI32); A(pI32); A(pI8);  A(pI32); break;
  case OpCode::LoadPatchConstant:      A(pETy);     A(pI32); A(pI32); A(pI32); A(pI8);  break;

    // Domain shader
  case OpCode::DomainLocation:         A(pF32);     A(pI32); A(pI8);  break;

    // Hull shader
  case OpCode::StorePatchConstant:     A(pV);       A(pI32); A(pI32); A(pI32); A(pI8);  A(pETy); break;
  case OpCode::OutputControlPointID:   A(pI32);     A(pI32); break;

    // Hull, Domain and Geometry shaders
  case OpCode::PrimitiveID:            A(pI32);     A(pI32); break;

    // Other
  case OpCode::CycleCounterLegacy:     A(p2I32);    A(pI32); break;

    // Wave
  case OpCode::WaveIsFirstLane:        A(pI1);      A(pI32); break;
  case OpCode::WaveGetLaneIndex:       A(pI32);     A(pI32); break;
  case OpCode::WaveGetLaneCount:       A(pI32);     A(pI32); break;
  case OpCode::WaveAnyTrue:            A(pI1);      A(pI32); A(pI1);  break;
  case OpCode::WaveAllTrue:            A(pI1);      A(pI32); A(pI1);  break;
  case OpCode::WaveActiveAllEqual:     A(pI1);      A(pI32); A(pETy); break;
  case OpCode::WaveActiveBallot:       A(p4I32);    A(pI32); A(pI1);  break;
  case OpCode::WaveReadLaneAt:         A(pETy);     A(pI32); A(pETy); A(pI32); break;
  case OpCode::WaveReadLaneFirst:      A(pETy);     A(pI32); A(pETy); break;
  case OpCode::WaveActiveOp:           A(pETy);     A(pI32); A(pETy); A(pI8);  A(pI8);  break;
  case OpCode::WaveActiveBit:          A(pETy);     A(pI32); A(pETy); A(pI8);  break;
  case OpCode::WavePrefixOp:           A(pETy);     A(pI32); A(pETy); A(pI8);  A(pI8);  break;

    // Quad Wave Ops
  case OpCode::QuadReadLaneAt:         A(pETy);     A(pI32); A(pETy); A(pI32); break;
  case OpCode::QuadOp:                 A(pETy);     A(pI32); A(pETy); A(pI8);  break;

    // Bitcasts with different sizes
  case OpCode::BitcastI16toF16:        A(pF16);     A(pI32); A(pI16); break;
  case OpCode::BitcastF16toI16:        A(pI16);     A(pI32); A(pF16); break;
  case OpCode::BitcastI32toF32:        A(pF32);     A(pI32); A(pI32); break;
  case OpCode::BitcastF32toI32:        A(pI32);     A(pI32); A(pF32); break;
  case OpCode::BitcastI64toF64:        A(pF64);     A(pI32); A(pI64); break;
  case OpCode::BitcastF64toI64:        A(pI64);     A(pI32); A(pF64); break;

    // Legacy floating-point
  case OpCode::LegacyF32ToF16:         A(pI32);     A(pI32); A(pF32); break;
  case OpCode::LegacyF16ToF32:         A(pF32);     A(pI32); A(pI32); break;

    // Double precision
  case OpCode::LegacyDoubleToFloat:    A(pF32);     A(pI32); A(pF64); break;
  case OpCode::LegacyDoubleToSInt32:   A(pI32);     A(pI32); A(pF64); break;
  case OpCode::LegacyDoubleToUInt32:   A(pI32);     A(pI32); A(pF64); break;

    // Wave
  case OpCode::WaveAllBitCount:        A(pI32);     A(pI32); A(pI1);  break;
  case OpCode::WavePrefixBitCount:     A(pI32);     A(pI32); A(pI1);  break;

    // Pixel shader
  case OpCode::AttributeAtVertex:      A(pETy);     A(pI32); A(pI32); A(pI32); A(pI8);  A(pI8);  break;

    // Graphics shader
  case OpCode::ViewID:                 A(pI32);     A(pI32); break;

    // Resources
  case OpCode::RawBufferLoad:          RRT(pETy);   A(pI32); A(pRes); A(pI32); A(pI32); A(pI8);  A(pI32); break;
  case OpCode::RawBufferStore:         A(pV);       A(pI32); A(pRes); A(pI32); A(pI32); A(pETy); A(pETy); A(pETy); A(pETy); A(pI8);  A(pI32); break;

    // Raytracing object space uint System Values
  case OpCode::InstanceID:             A(pI32);     A(pI32); break;
  case OpCode::InstanceIndex:          A(pI32);     A(pI32); break;

    // Raytracing hit uint System Values
  case OpCode::HitKind:                A(pI32);     A(pI32); break;

    // Raytracing uint System Values
  case OpCode::RayFlags:               A(pI32);     A(pI32); break;

    // Ray Dispatch Arguments
  case OpCode::DispatchRaysIndex:      A(pI32);     A(pI32); A(pI8);  break;
  case OpCode::DispatchRaysDimensions: A(pI32);     A(pI32); A(pI8);  break;

    // Ray Vectors
  case OpCode::WorldRayOrigin:         A(pF32);     A(pI32); A(pI8);  break;
  case OpCode::WorldRayDirection:      A(pF32);     A(pI32); A(pI8);  break;

    // Ray object space Vectors
  case OpCode::ObjectRayOrigin:        A(pF32);     A(pI32); A(pI8);  break;
  case OpCode::ObjectRayDirection:     A(pF32);     A(pI32); A(pI8);  break;

    // Ray Transforms
  case OpCode::ObjectToWorld:          A(pF32);     A(pI32); A(pI32); A(pI8);  break;
  case OpCode::WorldToObject:          A(pF32);     A(pI32); A(pI32); A(pI8);  break;

    // RayT
  case OpCode::RayTMin:                A(pF32);     A(pI32); break;
  case OpCode::RayTCurrent:            A(pF32);     A(pI32); break;

    // AnyHit Terminals
  case OpCode::IgnoreHit:              A(pV);       A(pI32); break;
  case OpCode::AcceptHitAndEndSearch:  A(pV);       A(pI32); break;

    // Indirect Shader Invocation
  case OpCode::TraceRay:               A(pV);       A(pI32); A(pRes); A(pI32); A(pI32); A(pI32); A(pI32); A(pI32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(udt);  break;
  case OpCode::ReportHit:              A(pI1);      A(pI32); A(pF32); A(pI32); A(udt);  break;
  case OpCode::CallShader:             A(pV);       A(pI32); A(pI32); A(udt);  break;

    // Library create handle from resource struct (like HL intrinsic)
  case OpCode::CreateHandleForLib:     A(pRes);     A(pI32); A(obj);  break;

    // Raytracing object space uint System Values
  case OpCode::PrimitiveIndex:         A(pI32);     A(pI32); break;

    // Dot product with accumulate
  case OpCode::Dot2AddHalf:            A(pETy);     A(pI32); A(pETy); A(pF16); A(pF16); A(pF16); A(pF16); break;
  case OpCode::Dot4AddI8Packed:        A(pI32);     A(pI32); A(pI32); A(pI32); A(pI32); break;
  case OpCode::Dot4AddU8Packed:        A(pI32);     A(pI32); A(pI32); A(pI32); A(pI32); break;

    // Wave
  case OpCode::WaveMatch:              A(p4I32);    A(pI32); A(pETy); break;
  case OpCode::WaveMultiPrefixOp:      A(pETy);     A(pI32); A(pETy); A(pI32); A(pI32); A(pI32); A(pI32); A(pI8);  A(pI8);  break;
  case OpCode::WaveMultiPrefixBitCount:A(pI32);     A(pI32); A(pI1);  A(pI32); A(pI32); A(pI32); A(pI32); break;

    // Mesh shader instructions
  case OpCode::SetMeshOutputCounts:    A(pV);       A(pI32); A(pI32); A(pI32); break;
  case OpCode::EmitIndices:            A(pV);       A(pI32); A(pI32); A(pI32); A(pI32); A(pI32); break;
  case OpCode::GetMeshPayload:         A(pETy);     A(pI32); break;
  case OpCode::StoreVertexOutput:      A(pV);       A(pI32); A(pI32); A(pI32); A(pI8);  A(pETy); A(pI32); break;
  case OpCode::StorePrimitiveOutput:   A(pV);       A(pI32); A(pI32); A(pI32); A(pI8);  A(pETy); A(pI32); break;

    // Amplification shader instructions
  case OpCode::DispatchMesh:           A(pV);       A(pI32); A(pI32); A(pI32); A(pI32); A(pETy); break;

    // Sampler Feedback
  case OpCode::WriteSamplerFeedback:   A(pV);       A(pI32); A(pRes); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); break;
  case OpCode::WriteSamplerFeedbackBias:A(pV);       A(pI32); A(pRes); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); break;
  case OpCode::WriteSamplerFeedbackLevel:A(pV);       A(pI32); A(pRes); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); break;
  case OpCode::WriteSamplerFeedbackGrad:A(pV);       A(pI32); A(pRes); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); break;

    // Inline Ray Query
  case OpCode::AllocateRayQuery:       A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_TraceRayInline:A(pV);       A(pI32); A(pI32); A(pRes); A(pI32); A(pI32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); A(pF32); break;
  case OpCode::RayQuery_Proceed:       A(pI1);      A(pI32); A(pI32); break;
  case OpCode::RayQuery_Abort:         A(pV);       A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommitNonOpaqueTriangleHit:A(pV);       A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommitProceduralPrimitiveHit:A(pV);       A(pI32); A(pI32); A(pF32); break;
  case OpCode::RayQuery_CommittedStatus:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateType: A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateObjectToWorld3x4:A(pF32);     A(pI32); A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_CandidateWorldToObject3x4:A(pF32);     A(pI32); A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_CommittedObjectToWorld3x4:A(pF32);     A(pI32); A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_CommittedWorldToObject3x4:A(pF32);     A(pI32); A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_CandidateProceduralPrimitiveNonOpaque:A(pI1);      A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateTriangleFrontFace:A(pI1);      A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommittedTriangleFrontFace:A(pI1);      A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateTriangleBarycentrics:A(pF32);     A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_CommittedTriangleBarycentrics:A(pF32);     A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_RayFlags:      A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_WorldRayOrigin:A(pF32);     A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_WorldRayDirection:A(pF32);     A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_RayTMin:       A(pF32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateTriangleRayT:A(pF32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommittedRayT: A(pF32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateInstanceIndex:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateInstanceID:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateGeometryIndex:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidatePrimitiveIndex:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CandidateObjectRayOrigin:A(pF32);     A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_CandidateObjectRayDirection:A(pF32);     A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_CommittedInstanceIndex:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommittedInstanceID:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommittedGeometryIndex:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommittedPrimitiveIndex:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommittedObjectRayOrigin:A(pF32);     A(pI32); A(pI32); A(pI8);  break;
  case OpCode::RayQuery_CommittedObjectRayDirection:A(pF32);     A(pI32); A(pI32); A(pI8);  break;

    // Raytracing object space uint System Values, raytracing tier 1.1
  case OpCode::GeometryIndex:          A(pI32);     A(pI32); break;

    // Inline Ray Query
  case OpCode::RayQuery_CandidateInstanceContributionToHitGroupIndex:A(pI32);     A(pI32); A(pI32); break;
  case OpCode::RayQuery_CommittedInstanceContributionToHitGroupIndex:A(pI32);     A(pI32); A(pI32); break;

    // Get handle from heap
  case OpCode::AnnotateHandle:         A(pRes);     A(pI32); A(pRes); A(resProperty);break;
  case OpCode::CreateHandleFromBinding:A(pRes);     A(pI32); A(resBind);A(pI32); A(pI1);  break;
  case OpCode::CreateHandleFromHeap:   A(pRes);     A(pI32); A(pI32); A(pI1);  A(pI1);  break;

    // Unpacking intrinsics
  case OpCode::Unpack4x8:              VEC4(pETy);  A(pI32); A(pI8);  A(pI32); break;

    // Packing intrinsics
  case OpCode::Pack4x8:                A(pI32);     A(pI32); A(pI8);  A(pETy); A(pETy); A(pETy); A(pETy); break;

    // Helper Lanes
  case OpCode::IsHelperLane:           A(pI1);      A(pI32); break;

    // Quad Wave Ops
  case OpCode::QuadVote:               A(pI1);      A(pI32); A(pI1);  A(pI8);  break;

    // Resources - gather
  case OpCode::TextureGatherRaw:       RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); break;

    // Resources - sample
  case OpCode::SampleCmpLevel:         RRT(pETy);   A(pI32); A(pRes); A(pRes); A(pF32); A(pF32); A(pF32); A(pF32); A(pI32); A(pI32); A(pI32); A(pF32); A(pF32); break;

    // Resources
  case OpCode::TextureStoreSample:     A(pV);       A(pI32); A(pRes); A(pI32); A(pI32); A(pI32); A(pETy); A(pETy); A(pETy); A(pETy); A(pI8);  A(pI32); break;
  // OPCODE-OLOAD-FUNCS:END
  default: DXASSERT(false, "otherwise unhandled case"); break;
  }
#undef RRT
#undef A

  FunctionType *pFT;
  DXASSERT(ArgTypes.size() > 1, "otherwise forgot to initialize arguments");
  pFT = FunctionType::get(ArgTypes[0], ArrayRef<Type*>(&ArgTypes[1], ArgTypes.size()-1), false);

  F = cast<Function>(m_pModule->getOrInsertFunction(funcName, pFT));

  UpdateCache(opClass, pOverloadType, F);
  F->setCallingConv(CallingConv::C);
  F->addFnAttr(Attribute::NoUnwind);
  if (m_OpCodeProps[(unsigned)opCode].FuncAttr != Attribute::None)
    F->addFnAttr(m_OpCodeProps[(unsigned)opCode].FuncAttr);

  return F;
}

const SmallMapVector<llvm::Type *, llvm::Function *, 8> &
OP::GetOpFuncList(OpCode opCode) const {
  DXASSERT(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes,
           "otherwise caller passed OOB OpCode");
  _Analysis_assume_(0 <= (unsigned)opCode && opCode < OpCode::NumOpCodes);
  return m_OpCodeClassCache[(unsigned)m_OpCodeProps[(unsigned)opCode]
                                .opCodeClass]
      .pOverloads;
}

bool OP::IsDxilOpUsed(OpCode opcode) const {
  auto &FnList = GetOpFuncList(opcode);
  for (auto &it : FnList) {
    llvm::Function *F = it.second;
    if (!F) {
      continue;
    }
    if (!F->user_empty()) {
      return true;
    }
  }
  return false;
}

void OP::RemoveFunction(Function *F) {
  if (OP::IsDxilOpFunc(F)) {
    OpCodeClass opClass = m_FunctionToOpClass[F];
    for (auto it : m_OpCodeClassCache[(unsigned)opClass].pOverloads) {
      if (it.second == F) {
        m_OpCodeClassCache[(unsigned)opClass].pOverloads.erase(it.first);
        m_FunctionToOpClass.erase(F);
        break;
      }
    }
  }
}

bool OP::GetOpCodeClass(const Function *F, OP::OpCodeClass &opClass) {
  auto iter = m_FunctionToOpClass.find(F);
  if (iter == m_FunctionToOpClass.end()) {
    // When no user, cannot get opcode.
    DXASSERT(F->user_empty() || !IsDxilOpFunc(F), "dxil function without an opcode class mapping?");
    opClass = OP::OpCodeClass::NumOpClasses;
    return false;
  }
  opClass = iter->second;
  return true;
}

bool OP::UseMinPrecision() {
  return m_LowPrecisionMode == DXIL::LowPrecisionMode::UseMinPrecision;
}

void OP::SetMinPrecision(bool bMinPrecision) {
  DXIL::LowPrecisionMode mode =
      bMinPrecision ? DXIL::LowPrecisionMode::UseMinPrecision
                    : DXIL::LowPrecisionMode::UseNativeLowPrecision;
  DXASSERT((mode == m_LowPrecisionMode ||
            m_LowPrecisionMode == DXIL::LowPrecisionMode::Undefined),
           "LowPrecisionMode should only be set once.");

  m_LowPrecisionMode = mode;
}

uint64_t OP::GetAllocSizeForType(llvm::Type *Ty) {
  return m_pModule->getDataLayout().getTypeAllocSize(Ty);
}

llvm::Type *OP::GetOverloadType(OpCode opCode, llvm::Function *F) {
  DXASSERT(F, "not work on nullptr");
  Type *Ty = F->getReturnType();
  FunctionType *FT = F->getFunctionType();
  LLVMContext &Ctx = F->getContext();
/* <py::lines('OPCODE-OLOAD-TYPES')>hctdb_instrhelp.get_funcs_oload_type()</py>*/
  switch (opCode) {            // return     OpCode
  // OPCODE-OLOAD-TYPES:BEGIN
  case OpCode::TempRegStore:
  case OpCode::CallShader:
  case OpCode::Pack4x8:
    DXASSERT_NOMSG(FT->getNumParams() > 2);
    return FT->getParamType(2);
  case OpCode::MinPrecXRegStore:
  case OpCode::StoreOutput:
  case OpCode::BufferStore:
  case OpCode::StorePatchConstant:
  case OpCode::RawBufferStore:
  case OpCode::StoreVertexOutput:
  case OpCode::StorePrimitiveOutput:
  case OpCode::DispatchMesh:
    DXASSERT_NOMSG(FT->getNumParams() > 4);
    return FT->getParamType(4);
  case OpCode::IsNaN:
  case OpCode::IsInf:
  case OpCode::IsFinite:
  case OpCode::IsNormal:
  case OpCode::Countbits:
  case OpCode::FirstbitLo:
  case OpCode::FirstbitHi:
  case OpCode::FirstbitSHi:
  case OpCode::IMul:
  case OpCode::UMul:
  case OpCode::UDiv:
  case OpCode::UAddc:
  case OpCode::USubb:
  case OpCode::WaveActiveAllEqual:
  case OpCode::CreateHandleForLib:
  case OpCode::WaveMatch:
    DXASSERT_NOMSG(FT->getNumParams() > 1);
    return FT->getParamType(1);
  case OpCode::TextureStore:
  case OpCode::TextureStoreSample:
    DXASSERT_NOMSG(FT->getNumParams() > 5);
    return FT->getParamType(5);
  case OpCode::TraceRay:
    DXASSERT_NOMSG(FT->getNumParams() > 15);
    return FT->getParamType(15);
  case OpCode::ReportHit:
    DXASSERT_NOMSG(FT->getNumParams() > 3);
    return FT->getParamType(3);
  case OpCode::CreateHandle:
  case OpCode::BufferUpdateCounter:
  case OpCode::GetDimensions:
  case OpCode::Texture2DMSGetSamplePosition:
  case OpCode::RenderTargetGetSamplePosition:
  case OpCode::RenderTargetGetSampleCount:
  case OpCode::Barrier:
  case OpCode::Discard:
  case OpCode::EmitStream:
  case OpCode::CutStream:
  case OpCode::EmitThenCutStream:
  case OpCode::CycleCounterLegacy:
  case OpCode::WaveIsFirstLane:
  case OpCode::WaveGetLaneIndex:
  case OpCode::WaveGetLaneCount:
  case OpCode::WaveAnyTrue:
  case OpCode::WaveAllTrue:
  case OpCode::WaveActiveBallot:
  case OpCode::BitcastI16toF16:
  case OpCode::BitcastF16toI16:
  case OpCode::BitcastI32toF32:
  case OpCode::BitcastF32toI32:
  case OpCode::BitcastI64toF64:
  case OpCode::BitcastF64toI64:
  case OpCode::LegacyF32ToF16:
  case OpCode::LegacyF16ToF32:
  case OpCode::LegacyDoubleToFloat:
  case OpCode::LegacyDoubleToSInt32:
  case OpCode::LegacyDoubleToUInt32:
  case OpCode::WaveAllBitCount:
  case OpCode::WavePrefixBitCount:
  case OpCode::IgnoreHit:
  case OpCode::AcceptHitAndEndSearch:
  case OpCode::WaveMultiPrefixBitCount:
  case OpCode::SetMeshOutputCounts:
  case OpCode::EmitIndices:
  case OpCode::WriteSamplerFeedback:
  case OpCode::WriteSamplerFeedbackBias:
  case OpCode::WriteSamplerFeedbackLevel:
  case OpCode::WriteSamplerFeedbackGrad:
  case OpCode::AllocateRayQuery:
  case OpCode::RayQuery_TraceRayInline:
  case OpCode::RayQuery_Abort:
  case OpCode::RayQuery_CommitNonOpaqueTriangleHit:
  case OpCode::RayQuery_CommitProceduralPrimitiveHit:
  case OpCode::AnnotateHandle:
  case OpCode::CreateHandleFromBinding:
  case OpCode::CreateHandleFromHeap:
    return Type::getVoidTy(Ctx);
  case OpCode::CheckAccessFullyMapped:
  case OpCode::SampleIndex:
  case OpCode::Coverage:
  case OpCode::InnerCoverage:
  case OpCode::ThreadId:
  case OpCode::GroupId:
  case OpCode::ThreadIdInGroup:
  case OpCode::FlattenedThreadIdInGroup:
  case OpCode::GSInstanceID:
  case OpCode::OutputControlPointID:
  case OpCode::PrimitiveID:
  case OpCode::ViewID:
  case OpCode::InstanceID:
  case OpCode::InstanceIndex:
  case OpCode::HitKind:
  case OpCode::RayFlags:
  case OpCode::DispatchRaysIndex:
  case OpCode::DispatchRaysDimensions:
  case OpCode::PrimitiveIndex:
  case OpCode::Dot4AddI8Packed:
  case OpCode::Dot4AddU8Packed:
  case OpCode::RayQuery_CommittedStatus:
  case OpCode::RayQuery_CandidateType:
  case OpCode::RayQuery_RayFlags:
  case OpCode::RayQuery_CandidateInstanceIndex:
  case OpCode::RayQuery_CandidateInstanceID:
  case OpCode::RayQuery_CandidateGeometryIndex:
  case OpCode::RayQuery_CandidatePrimitiveIndex:
  case OpCode::RayQuery_CommittedInstanceIndex:
  case OpCode::RayQuery_CommittedInstanceID:
  case OpCode::RayQuery_CommittedGeometryIndex:
  case OpCode::RayQuery_CommittedPrimitiveIndex:
  case OpCode::GeometryIndex:
  case OpCode::RayQuery_CandidateInstanceContributionToHitGroupIndex:
  case OpCode::RayQuery_CommittedInstanceContributionToHitGroupIndex:
    return IntegerType::get(Ctx, 32);
  case OpCode::CalculateLOD:
  case OpCode::DomainLocation:
  case OpCode::WorldRayOrigin:
  case OpCode::WorldRayDirection:
  case OpCode::ObjectRayOrigin:
  case OpCode::ObjectRayDirection:
  case OpCode::ObjectToWorld:
  case OpCode::WorldToObject:
  case OpCode::RayTMin:
  case OpCode::RayTCurrent:
  case OpCode::RayQuery_CandidateObjectToWorld3x4:
  case OpCode::RayQuery_CandidateWorldToObject3x4:
  case OpCode::RayQuery_CommittedObjectToWorld3x4:
  case OpCode::RayQuery_CommittedWorldToObject3x4:
  case OpCode::RayQuery_CandidateTriangleBarycentrics:
  case OpCode::RayQuery_CommittedTriangleBarycentrics:
  case OpCode::RayQuery_WorldRayOrigin:
  case OpCode::RayQuery_WorldRayDirection:
  case OpCode::RayQuery_RayTMin:
  case OpCode::RayQuery_CandidateTriangleRayT:
  case OpCode::RayQuery_CommittedRayT:
  case OpCode::RayQuery_CandidateObjectRayOrigin:
  case OpCode::RayQuery_CandidateObjectRayDirection:
  case OpCode::RayQuery_CommittedObjectRayOrigin:
  case OpCode::RayQuery_CommittedObjectRayDirection:
    return Type::getFloatTy(Ctx);
  case OpCode::MakeDouble:
  case OpCode::SplitDouble:
    return Type::getDoubleTy(Ctx);
  case OpCode::RayQuery_Proceed:
  case OpCode::RayQuery_CandidateProceduralPrimitiveNonOpaque:
  case OpCode::RayQuery_CandidateTriangleFrontFace:
  case OpCode::RayQuery_CommittedTriangleFrontFace:
  case OpCode::IsHelperLane:
  case OpCode::QuadVote:
    return IntegerType::get(Ctx, 1);
  case OpCode::CBufferLoadLegacy:
  case OpCode::Sample:
  case OpCode::SampleBias:
  case OpCode::SampleLevel:
  case OpCode::SampleGrad:
  case OpCode::SampleCmp:
  case OpCode::SampleCmpLevelZero:
  case OpCode::TextureLoad:
  case OpCode::BufferLoad:
  case OpCode::TextureGather:
  case OpCode::TextureGatherCmp:
  case OpCode::RawBufferLoad:
  case OpCode::Unpack4x8:
  case OpCode::TextureGatherRaw:
  case OpCode::SampleCmpLevel:
  {
    StructType *ST = cast<StructType>(Ty);
    return ST->getElementType(0);
  }
  // OPCODE-OLOAD-TYPES:END
  default: return Ty;
  }
}

Type *OP::GetHandleType() const {
  return m_pHandleType;
}

Type *OP::GetResourcePropertiesType() const {
  return m_pResourcePropertiesType;
}

Type *OP::GetResourceBindingType() const {
  return m_pResourceBindingType;
}

Type *OP::GetDimensionsType() const
{
  return m_pDimensionsType;
}

Type *OP::GetSamplePosType() const
{
  return m_pSamplePosType;
}

Type *OP::GetBinaryWithCarryType() const {
  return m_pBinaryWithCarryType;
}

Type *OP::GetBinaryWithTwoOutputsType() const {
  return m_pBinaryWithTwoOutputsType;
}

Type *OP::GetSplitDoubleType() const {
  return m_pSplitDoubleType;
}

Type *OP::GetFourI32Type() const {
  return m_pFourI32Type;
}

Type *OP::GetFourI16Type() const {
  return m_pFourI16Type;
}

bool OP::IsResRetType(llvm::Type *Ty) {
  for (Type *ResTy : m_pResRetType) {
    if (Ty == ResTy)
      return true;
  }
  return false;
}

Type *OP::GetResRetType(Type *pOverloadType) {
  unsigned TypeSlot = GetTypeSlot(pOverloadType);

  if (m_pResRetType[TypeSlot] == nullptr) {
    string TypeName("dx.types.ResRet.");
    TypeName += GetOverloadTypeName(TypeSlot);
    Type *FieldTypes[5] = { pOverloadType, pOverloadType, pOverloadType, pOverloadType, Type::getInt32Ty(m_Ctx) };
    m_pResRetType[TypeSlot] = GetOrCreateStructType(m_Ctx, FieldTypes, TypeName, m_pModule);
  }

  return m_pResRetType[TypeSlot];
}

Type *OP::GetCBufferRetType(Type *pOverloadType) {
  unsigned TypeSlot = GetTypeSlot(pOverloadType);

  if (m_pCBufferRetType[TypeSlot] == nullptr) {
    string TypeName("dx.types.CBufRet.");
    TypeName += GetOverloadTypeName(TypeSlot);
    Type *i64Ty = Type::getInt64Ty(pOverloadType->getContext());
    Type *i16Ty = Type::getInt16Ty(pOverloadType->getContext());
    if (pOverloadType->isDoubleTy() || pOverloadType == i64Ty) {
      Type *FieldTypes[2] = { pOverloadType, pOverloadType };
      m_pCBufferRetType[TypeSlot] = GetOrCreateStructType(m_Ctx, FieldTypes, TypeName, m_pModule);
    }
    else if (!UseMinPrecision() && (pOverloadType->isHalfTy() || pOverloadType == i16Ty)) {
      TypeName += ".8"; // dx.types.CBufRet.fp16.8 for buffer of 8 halves
      Type *FieldTypes[8] = {
          pOverloadType, pOverloadType, pOverloadType, pOverloadType,
          pOverloadType, pOverloadType, pOverloadType, pOverloadType,
      };
      m_pCBufferRetType[TypeSlot] = GetOrCreateStructType(m_Ctx, FieldTypes, TypeName, m_pModule);
    }
    else {
      Type *FieldTypes[4] = { pOverloadType, pOverloadType, pOverloadType, pOverloadType };
      m_pCBufferRetType[TypeSlot] = GetOrCreateStructType(m_Ctx, FieldTypes, TypeName, m_pModule);
    }
  }
  return m_pCBufferRetType[TypeSlot];
}

Type *OP::GetVectorType(unsigned numElements, Type *pOverloadType) {
  if (numElements == 4) {
    if (pOverloadType == Type::getInt32Ty(pOverloadType->getContext())) {
      return m_pFourI32Type;
    }
    else if (pOverloadType == Type::getInt16Ty(pOverloadType->getContext())) {
      return m_pFourI16Type;
    }
  }
  DXASSERT(false, "unexpected overload type");
  return nullptr;
}

//------------------------------------------------------------------------------
//
//  LLVM utility methods.
//
Constant *OP::GetI1Const(bool v) {
  return Constant::getIntegerValue(IntegerType::get(m_Ctx, 1), APInt(1, v));
}

Constant *OP::GetI8Const(char v) {
  return Constant::getIntegerValue(IntegerType::get(m_Ctx, 8), APInt(8, v));
}

Constant *OP::GetU8Const(unsigned char v) {
  return GetI8Const((char)v);
}

Constant *OP::GetI16Const(int v) {
  return Constant::getIntegerValue(IntegerType::get(m_Ctx, 16), APInt(16, v));
}

Constant *OP::GetU16Const(unsigned v) {
  return GetI16Const((int)v);
}

Constant *OP::GetI32Const(int v) {
  return Constant::getIntegerValue(IntegerType::get(m_Ctx, 32), APInt(32, v));
}

Constant *OP::GetU32Const(unsigned v) {
  return GetI32Const((int)v);
}

Constant *OP::GetU64Const(unsigned long long v) {
 return Constant::getIntegerValue(IntegerType::get(m_Ctx, 64), APInt(64, v));
}

Constant *OP::GetFloatConst(float v) {
  return ConstantFP::get(m_Ctx, APFloat(v));
}

Constant *OP::GetDoubleConst(double v) {
  return ConstantFP::get(m_Ctx, APFloat(v));
}

} // namespace hlsl
