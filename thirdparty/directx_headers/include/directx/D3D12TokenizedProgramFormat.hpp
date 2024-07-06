#pragma once
//*********************************************************
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// High Level Goals
//
// - Serve as the runtime/DDI representation for all D3D11 tokenized code,
//   for all classes of programs, including pixel program, vertex program,
//   geometry program, etc.
//
// - Any information that HLSL needs to give to drivers is encoded in
//   this token format in some form.
//
// - Enable common tools and source code for managing all tokenizable
//   program formats.
//
// - Support extensible token definitions, allowing full customizations for
//   specific program classes, while maintaining general conventions for all
//   program models.
//
// - Binary backwards compatible with D3D10.  Any token name that was originally
//   defined with "D3D10" in it is unchanged; D3D11 only adds new tokens.
//
// ----------------------------------------------------------------------------
//
// Low Level Feature Summary
//
// - DWORD based tokens always, for simplicity
// - Opcode token is generally a single DWORD, though there is a bit indicating
//   if extended information (extra DWORD(s)) are present
// - Operand tokens are a completely self contained, extensible format,
//   with scalar and 4-vector data types as first class citizens, but
//   allowance for extension to n-component vectors.
// - Initial operand token identifies register type, register file
//   structure/dimensionality and mode of indexing for each dimension,
//   and choice of component selection mechanism (i.e. mask vs. swizzle etc).
// - Optional additional extended operand tokens can defined things like
//   modifiers (which are not needed by default).
// - Operand's immediate index value(s), if needed, appear as subsequent DWORD
//   values, and if relative addressing is specified, an additional completely
//   self contained operand definition appears nested in the token sequence.
//
// ----------------------------------------------------------------------------

#include <winapifamily.h>

#pragma region Application Family
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES)

// ----------------------------------------------------------------------------
// Version Token (VerTok)
//
// [07:00] minor version number (0-255)
// [15:08] major version number (0-255)
// [31:16] D3D10_SB_TOKENIZED_PROGRAM_TYPE
//
// ----------------------------------------------------------------------------

typedef enum D3D10_SB_TOKENIZED_PROGRAM_TYPE
{
    D3D10_SB_PIXEL_SHADER       = 0,
    D3D10_SB_VERTEX_SHADER      = 1,
    D3D10_SB_GEOMETRY_SHADER    = 2,
    
    // D3D11 Shaders
    D3D11_SB_HULL_SHADER        = 3,
    D3D11_SB_DOMAIN_SHADER      = 4,
    D3D11_SB_COMPUTE_SHADER     = 5,

    // Subset of D3D12 Shaders where this field is referenced by runtime
    // Entries from 6-12 are unique to state objects 
    // (e.g. library, callable and raytracing shaders)
    D3D12_SB_MESH_SHADER        = 13,
    D3D12_SB_AMPLIFICATION_SHADER = 14,

    D3D11_SB_RESERVED0          = 0xFFF0
} D3D10_SB_TOKENIZED_PROGRAM_TYPE;

#define D3D10_SB_TOKENIZED_PROGRAM_TYPE_MASK  0xffff0000
#define D3D10_SB_TOKENIZED_PROGRAM_TYPE_SHIFT 16

// DECODER MACRO: Retrieve program type from version token
#define DECODE_D3D10_SB_TOKENIZED_PROGRAM_TYPE(VerTok) ((D3D10_SB_TOKENIZED_PROGRAM_TYPE)(((VerTok)&D3D10_SB_TOKENIZED_PROGRAM_TYPE_MASK)>>D3D10_SB_TOKENIZED_PROGRAM_TYPE_SHIFT))

#define D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION_MASK  0x000000f0
#define D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION_SHIFT 4
#define D3D10_SB_TOKENIZED_PROGRAM_MINOR_VERSION_MASK  0x0000000f

// DECODER MACRO: Retrieve major version # from version token
#define DECODE_D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION(VerTok) (((VerTok)&D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION_MASK)>>D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION_SHIFT)
// DECODER MACRO: Retrieve minor version # from version token
#define DECODE_D3D10_SB_TOKENIZED_PROGRAM_MINOR_VERSION(VerTok) ((VerTok)&D3D10_SB_TOKENIZED_PROGRAM_MINOR_VERSION_MASK)

// ENCODER MACRO: Create complete VerTok
#define ENCODE_D3D10_SB_TOKENIZED_PROGRAM_VERSION_TOKEN(ProgType,MajorVer,MinorVer) ((((ProgType)<<D3D10_SB_TOKENIZED_PROGRAM_TYPE_SHIFT)&D3D10_SB_TOKENIZED_PROGRAM_TYPE_MASK)|\
                                                                               ((((MajorVer)<<D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION_SHIFT)&D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION_MASK))|\
                                                                               ((MinorVer)&D3D10_SB_TOKENIZED_PROGRAM_MINOR_VERSION_MASK))

// ----------------------------------------------------------------------------
// Length Token (LenTok)
//
// Always follows VerTok
//
// [31:00] Unsigned integer count of number of
//              DWORDs in program code, including version
//              and length tokens.  So the minimum value
//              is 0x00000002 (if an empty program is ever
//              valid).
//
// ----------------------------------------------------------------------------

// DECODER MACRO: Retrieve program length
#define DECODE_D3D10_SB_TOKENIZED_PROGRAM_LENGTH(LenTok) (LenTok)
// ENCODER MACRO: Create complete LenTok
#define ENCODE_D3D10_SB_TOKENIZED_PROGRAM_LENGTH(Length) (Length)
#define MAX_D3D10_SB_TOKENIZED_PROGRAM_LENGTH (0xffffffff)

// ----------------------------------------------------------------------------
// Opcode Format (OpcodeToken0)
//
// [10:00] D3D10_SB_OPCODE_TYPE
// if( [10:00] == D3D10_SB_OPCODE_CUSTOMDATA )
// {
//    Token starts a custom-data block.  See "Custom-Data Block Format".
// }
// else // standard opcode token
// {
//    [23:11] Opcode-Specific Controls
//    [30:24] Instruction length in DWORDs including the opcode token.
//    [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//            contains extended opcode token.
// }
//
// ----------------------------------------------------------------------------

typedef enum D3D10_SB_OPCODE_TYPE {
    D3D10_SB_OPCODE_ADD          ,
    D3D10_SB_OPCODE_AND          ,
    D3D10_SB_OPCODE_BREAK        ,
    D3D10_SB_OPCODE_BREAKC       ,
    D3D10_SB_OPCODE_CALL         ,
    D3D10_SB_OPCODE_CALLC        ,
    D3D10_SB_OPCODE_CASE         ,
    D3D10_SB_OPCODE_CONTINUE     ,
    D3D10_SB_OPCODE_CONTINUEC    ,
    D3D10_SB_OPCODE_CUT          ,
    D3D10_SB_OPCODE_DEFAULT      ,
    D3D10_SB_OPCODE_DERIV_RTX    ,
    D3D10_SB_OPCODE_DERIV_RTY    ,
    D3D10_SB_OPCODE_DISCARD      ,
    D3D10_SB_OPCODE_DIV          ,
    D3D10_SB_OPCODE_DP2          ,
    D3D10_SB_OPCODE_DP3          ,
    D3D10_SB_OPCODE_DP4          ,
    D3D10_SB_OPCODE_ELSE         ,
    D3D10_SB_OPCODE_EMIT         ,
    D3D10_SB_OPCODE_EMITTHENCUT  ,
    D3D10_SB_OPCODE_ENDIF        ,
    D3D10_SB_OPCODE_ENDLOOP      ,
    D3D10_SB_OPCODE_ENDSWITCH    ,
    D3D10_SB_OPCODE_EQ           ,
    D3D10_SB_OPCODE_EXP          ,
    D3D10_SB_OPCODE_FRC          ,
    D3D10_SB_OPCODE_FTOI         ,
    D3D10_SB_OPCODE_FTOU         ,
    D3D10_SB_OPCODE_GE           ,
    D3D10_SB_OPCODE_IADD         ,
    D3D10_SB_OPCODE_IF           ,
    D3D10_SB_OPCODE_IEQ          ,
    D3D10_SB_OPCODE_IGE          ,
    D3D10_SB_OPCODE_ILT          ,
    D3D10_SB_OPCODE_IMAD         ,
    D3D10_SB_OPCODE_IMAX         ,
    D3D10_SB_OPCODE_IMIN         ,
    D3D10_SB_OPCODE_IMUL         ,
    D3D10_SB_OPCODE_INE          ,
    D3D10_SB_OPCODE_INEG         ,
    D3D10_SB_OPCODE_ISHL         ,
    D3D10_SB_OPCODE_ISHR         ,
    D3D10_SB_OPCODE_ITOF         ,
    D3D10_SB_OPCODE_LABEL        ,
    D3D10_SB_OPCODE_LD           ,
    D3D10_SB_OPCODE_LD_MS        ,
    D3D10_SB_OPCODE_LOG          ,
    D3D10_SB_OPCODE_LOOP         ,
    D3D10_SB_OPCODE_LT           ,
    D3D10_SB_OPCODE_MAD          ,
    D3D10_SB_OPCODE_MIN          ,
    D3D10_SB_OPCODE_MAX          ,
    D3D10_SB_OPCODE_CUSTOMDATA   ,
    D3D10_SB_OPCODE_MOV          ,
    D3D10_SB_OPCODE_MOVC         ,
    D3D10_SB_OPCODE_MUL          ,
    D3D10_SB_OPCODE_NE           ,
    D3D10_SB_OPCODE_NOP          ,
    D3D10_SB_OPCODE_NOT          ,
    D3D10_SB_OPCODE_OR           ,
    D3D10_SB_OPCODE_RESINFO      ,
    D3D10_SB_OPCODE_RET          ,
    D3D10_SB_OPCODE_RETC         ,
    D3D10_SB_OPCODE_ROUND_NE     ,
    D3D10_SB_OPCODE_ROUND_NI     ,
    D3D10_SB_OPCODE_ROUND_PI     ,
    D3D10_SB_OPCODE_ROUND_Z      ,
    D3D10_SB_OPCODE_RSQ          ,
    D3D10_SB_OPCODE_SAMPLE       ,
    D3D10_SB_OPCODE_SAMPLE_C     ,
    D3D10_SB_OPCODE_SAMPLE_C_LZ  ,
    D3D10_SB_OPCODE_SAMPLE_L     ,
    D3D10_SB_OPCODE_SAMPLE_D     ,
    D3D10_SB_OPCODE_SAMPLE_B     ,
    D3D10_SB_OPCODE_SQRT         ,
    D3D10_SB_OPCODE_SWITCH       ,
    D3D10_SB_OPCODE_SINCOS       ,
    D3D10_SB_OPCODE_UDIV         ,
    D3D10_SB_OPCODE_ULT          ,
    D3D10_SB_OPCODE_UGE          ,
    D3D10_SB_OPCODE_UMUL         ,
    D3D10_SB_OPCODE_UMAD         ,
    D3D10_SB_OPCODE_UMAX         ,
    D3D10_SB_OPCODE_UMIN         ,
    D3D10_SB_OPCODE_USHR         ,
    D3D10_SB_OPCODE_UTOF         ,
    D3D10_SB_OPCODE_XOR          ,
    D3D10_SB_OPCODE_DCL_RESOURCE                     , // DCL* opcodes have
    D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER              , // custom operand formats.
    D3D10_SB_OPCODE_DCL_SAMPLER                      ,
    D3D10_SB_OPCODE_DCL_INDEX_RANGE                  ,
    D3D10_SB_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY ,
    D3D10_SB_OPCODE_DCL_GS_INPUT_PRIMITIVE           ,
    D3D10_SB_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT      ,
    D3D10_SB_OPCODE_DCL_INPUT                        ,
    D3D10_SB_OPCODE_DCL_INPUT_SGV                    ,
    D3D10_SB_OPCODE_DCL_INPUT_SIV                    ,
    D3D10_SB_OPCODE_DCL_INPUT_PS                     ,
    D3D10_SB_OPCODE_DCL_INPUT_PS_SGV                 ,
    D3D10_SB_OPCODE_DCL_INPUT_PS_SIV                 ,
    D3D10_SB_OPCODE_DCL_OUTPUT                       ,
    D3D10_SB_OPCODE_DCL_OUTPUT_SGV                   ,
    D3D10_SB_OPCODE_DCL_OUTPUT_SIV                   ,
    D3D10_SB_OPCODE_DCL_TEMPS                        ,
    D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP               ,
    D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS                 ,

// -----------------------------------------------

    // This marks the end of D3D10.0 opcodes
    D3D10_SB_OPCODE_RESERVED0,
    
// ---------- DX 10.1 op codes---------------------

    D3D10_1_SB_OPCODE_LOD,
    D3D10_1_SB_OPCODE_GATHER4,
    D3D10_1_SB_OPCODE_SAMPLE_POS,
    D3D10_1_SB_OPCODE_SAMPLE_INFO,

// -----------------------------------------------

    // This marks the end of D3D10.1 opcodes
    D3D10_1_SB_OPCODE_RESERVED1,

// ---------- DX 11 op codes---------------------
    D3D11_SB_OPCODE_HS_DECLS                         , // token marks beginning of HS sub-shader
    D3D11_SB_OPCODE_HS_CONTROL_POINT_PHASE           , // token marks beginning of HS sub-shader
    D3D11_SB_OPCODE_HS_FORK_PHASE                    , // token marks beginning of HS sub-shader
    D3D11_SB_OPCODE_HS_JOIN_PHASE                    , // token marks beginning of HS sub-shader

    D3D11_SB_OPCODE_EMIT_STREAM                      ,
    D3D11_SB_OPCODE_CUT_STREAM                       ,
    D3D11_SB_OPCODE_EMITTHENCUT_STREAM               ,
    D3D11_SB_OPCODE_INTERFACE_CALL                   ,

    D3D11_SB_OPCODE_BUFINFO                          ,
    D3D11_SB_OPCODE_DERIV_RTX_COARSE                 ,
    D3D11_SB_OPCODE_DERIV_RTX_FINE                   ,
    D3D11_SB_OPCODE_DERIV_RTY_COARSE                 ,
    D3D11_SB_OPCODE_DERIV_RTY_FINE                   ,
    D3D11_SB_OPCODE_GATHER4_C                        ,
    D3D11_SB_OPCODE_GATHER4_PO                       ,
    D3D11_SB_OPCODE_GATHER4_PO_C                     ,
    D3D11_SB_OPCODE_RCP                              ,
    D3D11_SB_OPCODE_F32TOF16                         ,
    D3D11_SB_OPCODE_F16TOF32                         ,
    D3D11_SB_OPCODE_UADDC                            ,
    D3D11_SB_OPCODE_USUBB                            ,
    D3D11_SB_OPCODE_COUNTBITS                        ,
    D3D11_SB_OPCODE_FIRSTBIT_HI                      ,
    D3D11_SB_OPCODE_FIRSTBIT_LO                      ,
    D3D11_SB_OPCODE_FIRSTBIT_SHI                     ,
    D3D11_SB_OPCODE_UBFE                             ,
    D3D11_SB_OPCODE_IBFE                             ,
    D3D11_SB_OPCODE_BFI                              ,
    D3D11_SB_OPCODE_BFREV                            ,
    D3D11_SB_OPCODE_SWAPC                            ,

    D3D11_SB_OPCODE_DCL_STREAM                       ,
    D3D11_SB_OPCODE_DCL_FUNCTION_BODY                ,
    D3D11_SB_OPCODE_DCL_FUNCTION_TABLE               ,
    D3D11_SB_OPCODE_DCL_INTERFACE                    ,
    
    D3D11_SB_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT    ,
    D3D11_SB_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT   ,
    D3D11_SB_OPCODE_DCL_TESS_DOMAIN                  ,
    D3D11_SB_OPCODE_DCL_TESS_PARTITIONING            ,
    D3D11_SB_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE        ,
    D3D11_SB_OPCODE_DCL_HS_MAX_TESSFACTOR            ,
    D3D11_SB_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT ,
    D3D11_SB_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT ,

    D3D11_SB_OPCODE_DCL_THREAD_GROUP                 ,
    D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED  ,
    D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_RAW    ,
    D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_STRUCTURED,
    D3D11_SB_OPCODE_DCL_THREAD_GROUP_SHARED_MEMORY_RAW,
    D3D11_SB_OPCODE_DCL_THREAD_GROUP_SHARED_MEMORY_STRUCTURED,
    D3D11_SB_OPCODE_DCL_RESOURCE_RAW                 ,
    D3D11_SB_OPCODE_DCL_RESOURCE_STRUCTURED          ,
    D3D11_SB_OPCODE_LD_UAV_TYPED                     ,
    D3D11_SB_OPCODE_STORE_UAV_TYPED                  ,
    D3D11_SB_OPCODE_LD_RAW                           ,
    D3D11_SB_OPCODE_STORE_RAW                        ,
    D3D11_SB_OPCODE_LD_STRUCTURED                    ,
    D3D11_SB_OPCODE_STORE_STRUCTURED                 ,
    D3D11_SB_OPCODE_ATOMIC_AND                       ,
    D3D11_SB_OPCODE_ATOMIC_OR                        ,
    D3D11_SB_OPCODE_ATOMIC_XOR                       ,
    D3D11_SB_OPCODE_ATOMIC_CMP_STORE                 ,
    D3D11_SB_OPCODE_ATOMIC_IADD                      ,
    D3D11_SB_OPCODE_ATOMIC_IMAX                      ,
    D3D11_SB_OPCODE_ATOMIC_IMIN                      ,
    D3D11_SB_OPCODE_ATOMIC_UMAX                      ,
    D3D11_SB_OPCODE_ATOMIC_UMIN                      ,
    D3D11_SB_OPCODE_IMM_ATOMIC_ALLOC                 ,
    D3D11_SB_OPCODE_IMM_ATOMIC_CONSUME               ,
    D3D11_SB_OPCODE_IMM_ATOMIC_IADD                  ,
    D3D11_SB_OPCODE_IMM_ATOMIC_AND                   ,
    D3D11_SB_OPCODE_IMM_ATOMIC_OR                    ,
    D3D11_SB_OPCODE_IMM_ATOMIC_XOR                   ,
    D3D11_SB_OPCODE_IMM_ATOMIC_EXCH                  ,
    D3D11_SB_OPCODE_IMM_ATOMIC_CMP_EXCH              ,
    D3D11_SB_OPCODE_IMM_ATOMIC_IMAX                  ,
    D3D11_SB_OPCODE_IMM_ATOMIC_IMIN                  ,
    D3D11_SB_OPCODE_IMM_ATOMIC_UMAX                  ,
    D3D11_SB_OPCODE_IMM_ATOMIC_UMIN                  ,   
    D3D11_SB_OPCODE_SYNC                             ,
    
    D3D11_SB_OPCODE_DADD                             ,
    D3D11_SB_OPCODE_DMAX                             ,
    D3D11_SB_OPCODE_DMIN                             ,
    D3D11_SB_OPCODE_DMUL                             ,
    D3D11_SB_OPCODE_DEQ                              ,
    D3D11_SB_OPCODE_DGE                              ,
    D3D11_SB_OPCODE_DLT                              ,
    D3D11_SB_OPCODE_DNE                              ,
    D3D11_SB_OPCODE_DMOV                             ,
    D3D11_SB_OPCODE_DMOVC                            ,
    D3D11_SB_OPCODE_DTOF                             ,
    D3D11_SB_OPCODE_FTOD                             ,

    D3D11_SB_OPCODE_EVAL_SNAPPED                     ,
    D3D11_SB_OPCODE_EVAL_SAMPLE_INDEX                ,
    D3D11_SB_OPCODE_EVAL_CENTROID                    ,
    
    D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT            ,

    D3D11_SB_OPCODE_ABORT                            ,
    D3D11_SB_OPCODE_DEBUG_BREAK                      ,

// -----------------------------------------------

    // This marks the end of D3D11.0 opcodes
    D3D11_SB_OPCODE_RESERVED0,

    D3D11_1_SB_OPCODE_DDIV,
    D3D11_1_SB_OPCODE_DFMA,
    D3D11_1_SB_OPCODE_DRCP,

    D3D11_1_SB_OPCODE_MSAD,

    D3D11_1_SB_OPCODE_DTOI,
    D3D11_1_SB_OPCODE_DTOU,
    D3D11_1_SB_OPCODE_ITOD,
    D3D11_1_SB_OPCODE_UTOD,

// -----------------------------------------------

    // This marks the end of D3D11.1 opcodes
    D3D11_1_SB_OPCODE_RESERVED0,

    D3DWDDM1_3_SB_OPCODE_GATHER4_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_GATHER4_C_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_GATHER4_PO_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_GATHER4_PO_C_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_LD_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_LD_MS_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_LD_UAV_TYPED_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_LD_RAW_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_LD_STRUCTURED_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_SAMPLE_L_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_SAMPLE_C_LZ_FEEDBACK,

    D3DWDDM1_3_SB_OPCODE_SAMPLE_CLAMP_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_SAMPLE_B_CLAMP_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_SAMPLE_D_CLAMP_FEEDBACK,
    D3DWDDM1_3_SB_OPCODE_SAMPLE_C_CLAMP_FEEDBACK,

    D3DWDDM1_3_SB_OPCODE_CHECK_ACCESS_FULLY_MAPPED,

// -----------------------------------------------

    // This marks the end of WDDM 1.3 opcodes
    D3DWDDM1_3_SB_OPCODE_RESERVED0,

    D3D10_SB_NUM_OPCODES                                     // Should be the last entry
} D3D10_SB_OPCODE_TYPE;

#define D3D10_SB_OPCODE_TYPE_MASK 0x00007ff
// DECODER MACRO: Retrieve program opcode
#define DECODE_D3D10_SB_OPCODE_TYPE(OpcodeToken0) ((D3D10_SB_OPCODE_TYPE)((OpcodeToken0)&D3D10_SB_OPCODE_TYPE_MASK))
// ENCODER MACRO: Create the opcode-type portion of OpcodeToken0
#define ENCODE_D3D10_SB_OPCODE_TYPE(OpcodeName) ((OpcodeName)&D3D10_SB_OPCODE_TYPE_MASK)

#define D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH_MASK 0x7f000000
#define D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH_SHIFT 24
// DECODER MACRO: Retrieve instruction length
// in # of DWORDs including the opcode token(s).
// The range is 1-127.
#define DECODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(OpcodeToken0) (((OpcodeToken0)&D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH_MASK)>> D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH_SHIFT)

// ENCODER MACRO: Store instruction length
// portion of OpcodeToken0, in # of DWORDs
// including the opcode token(s).
// Valid range is 1-127.
#define ENCODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(Length) (((Length)<<D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH_SHIFT)&D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH_MASK)
#define MAX_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH 127

#define D3D10_SB_INSTRUCTION_SATURATE_MASK 0x00002000
// DECODER MACRO: Check OpcodeToken0 to see if an instruction
// is to saturate the result [0..1]
// This flag is indicated by one of the bits in the
// opcode specific control range.
#define DECODE_IS_D3D10_SB_INSTRUCTION_SATURATE_ENABLED(OpcodeToken0) ((OpcodeToken0)&D3D10_SB_INSTRUCTION_SATURATE_MASK)
// ENCODER MACRO: Encode in OpcodeToken0 if instruction is to saturate the result.
#define ENCODE_D3D10_SB_INSTRUCTION_SATURATE(bSat) (((bSat)!=0)?D3D10_SB_INSTRUCTION_SATURATE_MASK:0)

// Boolean test for conditional instructions such as if (if_z or if_nz)
// This is part of the opcode specific control range.
typedef enum D3D10_SB_INSTRUCTION_TEST_BOOLEAN
{
    D3D10_SB_INSTRUCTION_TEST_ZERO       = 0,
    D3D10_SB_INSTRUCTION_TEST_NONZERO    = 1
} D3D10_SB_INSTRUCTION_TEST_BOOLEAN;
#define D3D10_SB_INSTRUCTION_TEST_BOOLEAN_MASK  0x00040000
#define D3D10_SB_INSTRUCTION_TEST_BOOLEAN_SHIFT 18

// DECODER MACRO: For an OpcodeToken0 for requires either a
// zero or non-zero test, determine which test was chosen.
#define DECODE_D3D10_SB_INSTRUCTION_TEST_BOOLEAN(OpcodeToken0) ((D3D10_SB_INSTRUCTION_TEST_BOOLEAN)(((OpcodeToken0)&D3D10_SB_INSTRUCTION_TEST_BOOLEAN_MASK)>>D3D10_SB_INSTRUCTION_TEST_BOOLEAN_SHIFT))
// ENCODER MACRO: Store "zero" or "nonzero" in the opcode
// specific control range of OpcodeToken0
#define ENCODE_D3D10_SB_INSTRUCTION_TEST_BOOLEAN(Boolean) (((Boolean)<<D3D10_SB_INSTRUCTION_TEST_BOOLEAN_SHIFT)&D3D10_SB_INSTRUCTION_TEST_BOOLEAN_MASK)

// Precise value mask (bits 19-22)
// This is part of the opcode specific control range.
// It's 1 bit per-channel of the output, for instructions with multiple
// output operands, it applies to that component in each operand. This
// uses the components defined in D3D10_SB_COMPONENT_NAME.
#define D3D11_SB_INSTRUCTION_PRECISE_VALUES_MASK  0x00780000
#define D3D11_SB_INSTRUCTION_PRECISE_VALUES_SHIFT 19

// DECODER MACRO: this macro extracts from OpcodeToken0 the 4 component
// (xyzw) mask, as a field of D3D10_SB_4_COMPONENT_[X|Y|Z|W] flags.
#define DECODE_D3D11_SB_INSTRUCTION_PRECISE_VALUES(OpcodeToken0) ((((OpcodeToken0)&D3D11_SB_INSTRUCTION_PRECISE_VALUES_MASK)>>D3D11_SB_INSTRUCTION_PRECISE_VALUES_SHIFT))
// ENCODER MACRO: Given a set of
// D3D10_SB_OPERAND_4_COMPONENT_[X|Y|Z|W] values
// or'd together, encode them in OpcodeToken0.
#define ENCODE_D3D11_SB_INSTRUCTION_PRECISE_VALUES(ComponentMask) (((ComponentMask)<<D3D11_SB_INSTRUCTION_PRECISE_VALUES_SHIFT)&D3D11_SB_INSTRUCTION_PRECISE_VALUES_MASK)

// resinfo instruction return type
typedef enum D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE
{
    D3D10_SB_RESINFO_INSTRUCTION_RETURN_FLOAT      = 0,
    D3D10_SB_RESINFO_INSTRUCTION_RETURN_RCPFLOAT   = 1,
    D3D10_SB_RESINFO_INSTRUCTION_RETURN_UINT       = 2
} D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE;

#define D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE_MASK  0x00001800
#define D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE_SHIFT 11

// DECODER MACRO: For an OpcodeToken0 for the resinfo instruction, 
// determine the return type.
#define DECODE_D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE(OpcodeToken0) ((D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE)(((OpcodeToken0)&D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE_MASK)>>D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE_SHIFT))
// ENCODER MACRO: Encode the return type for the resinfo instruction
// in the opcode specific control range of OpcodeToken0
#define ENCODE_D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE(ReturnType) (((ReturnType)<<D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE_SHIFT)&D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE_MASK)

// sync instruction flags
#define D3D11_SB_SYNC_THREADS_IN_GROUP              0x00000800
#define D3D11_SB_SYNC_THREAD_GROUP_SHARED_MEMORY    0x00001000
#define D3D11_SB_SYNC_UNORDERED_ACCESS_VIEW_MEMORY_GROUP  0x00002000
#define D3D11_SB_SYNC_UNORDERED_ACCESS_VIEW_MEMORY_GLOBAL  0x00004000
#define D3D11_SB_SYNC_FLAGS_MASK                    0x00007800

// DECODER MACRO: Retrieve flags for sync instruction from OpcodeToken0.
#define DECODE_D3D11_SB_SYNC_FLAGS(OperandToken0) ((OperandToken0)&D3D11_SB_SYNC_FLAGS_MASK)

// ENCODER MACRO: Given a set of sync instruciton flags, encode them in OpcodeToken0.
#define ENCODE_D3D11_SB_SYNC_FLAGS(Flags) ((Flags)&D3D11_SB_SYNC_FLAGS_MASK)

#define D3D10_SB_OPCODE_EXTENDED_MASK 0x80000000
#define D3D10_SB_OPCODE_EXTENDED_SHIFT 31
// DECODER MACRO: Determine if the opcode is extended
// by an additional opcode token.  Currently there are no
// extended opcodes.
#define DECODE_IS_D3D10_SB_OPCODE_EXTENDED(OpcodeToken0) (((OpcodeToken0)&D3D10_SB_OPCODE_EXTENDED_MASK)>> D3D10_SB_OPCODE_EXTENDED_SHIFT)
// ENCODER MACRO: Store in OpcodeToken0 whether the opcode is extended
// by an additional opcode token.  
#define ENCODE_D3D10_SB_OPCODE_EXTENDED(bExtended) (((bExtended)!=0)?D3D10_SB_OPCODE_EXTENDED_MASK:0)

// ----------------------------------------------------------------------------
// Extended Opcode Format (OpcodeToken1)
//
// If bit31 of an opcode token is set, the
// opcode has an additional extended opcode token DWORD
// directly following OpcodeToken0.  Other tokens
// expected for the opcode, such as the operand
// token(s) always follow
// OpcodeToken0 AND OpcodeToken1..n (extended
// opcode tokens, if present).
//
// [05:00] D3D10_SB_EXTENDED_OPCODE_TYPE
// [30:06] if([05:00] == D3D10_SB_EXTENDED_OPCODE_SAMPLE_CONTROLS)
//         {
//              This custom opcode contains controls for SAMPLE.
//              [08:06] Ignored, 0.
//              [12:09] U texel immediate offset (4 bit 2's comp) (0 default)
//              [16:13] V texel immediate offset (4 bit 2's comp) (0 default)
//              [20:17] W texel immediate offset (4 bit 2's comp) (0 default)
//              [30:14] Ignored, 0.
//         }
//         else if( [05:00] == D3D11_SB_EXTENDED_OPCODE_RESOURCE_DIM )
//         {
//              [10:06] D3D10_SB_RESOURCE_DIMENSION
//              [22:11] When dimension is D3D11_SB_RESOURCE_DIMENSION_STRUCTURED_BUFFER this holds the buffer stride, otherwise 0
//              [30:23] Ignored, 0.
//         }
//         else if( [05:00] == D3D11_SB_EXTENDED_OPCODE_RESOURCE_RETURN_TYPE )
//         {
//              [09:06] D3D10_SB_RESOURCE_RETURN_TYPE for component X
//              [13:10] D3D10_SB_RESOURCE_RETURN_TYPE for component Y
//              [17:14] D3D10_SB_RESOURCE_RETURN_TYPE for component Z
//              [21:18] D3D10_SB_RESOURCE_RETURN_TYPE for component W
//              [30:22] Ignored, 0.
//         }
//         else
//         {
//              [30:04] Ignored, 0.
//         }
// [31]    0 normally. 1 there is another extended opcode.  Any number
//         of extended opcode tokens can be chained.  It is possible that some extended
//         opcode tokens could include multiple DWORDS - that is defined
//         on a case by case basis.
//
// ----------------------------------------------------------------------------
typedef enum D3D10_SB_EXTENDED_OPCODE_TYPE
{
    D3D10_SB_EXTENDED_OPCODE_EMPTY           = 0,
    D3D10_SB_EXTENDED_OPCODE_SAMPLE_CONTROLS = 1,
    D3D11_SB_EXTENDED_OPCODE_RESOURCE_DIM = 2,
    D3D11_SB_EXTENDED_OPCODE_RESOURCE_RETURN_TYPE = 3,
} D3D10_SB_EXTENDED_OPCODE_TYPE;
#define D3D11_SB_MAX_SIMULTANEOUS_EXTENDED_OPCODES 3

#define D3D10_SB_EXTENDED_OPCODE_TYPE_MASK 0x0000003f

// DECODER MACRO: Given an extended opcode
// token (OpcodeToken1), figure out what type
// of token it is (from D3D10_SB_EXTENDED_OPCODE_TYPE enum)
// to be able to interpret the rest of the token's contents.
#define DECODE_D3D10_SB_EXTENDED_OPCODE_TYPE(OpcodeToken1) ((D3D10_SB_EXTENDED_OPCODE_TYPE)((OpcodeToken1)&D3D10_SB_EXTENDED_OPCODE_TYPE_MASK))

// ENCODER MACRO: Store extended opcode token
// type in OpcodeToken1.
#define ENCODE_D3D10_SB_EXTENDED_OPCODE_TYPE(ExtOpcodeType) ((ExtOpcodeType)&D3D10_SB_EXTENDED_OPCODE_TYPE_MASK)

typedef enum D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_COORD
{
    D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_U        = 0,
    D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_V        = 1,
    D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_W        = 2,
} D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_COORD;
#define D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_COORD_MASK (3)
#define D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_SHIFT(Coord) (9+4*((Coord)&D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_COORD_MASK))
#define D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_MASK(Coord) (0x0000000f<<D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_SHIFT(Coord))

// DECODER MACRO: Given an extended opcode token
// (OpcodeToken1), and extended token type ==
// D3D10_SB_EXTENDED_OPCODE_SAMPLE_CONTROLS, determine the immediate
// texel address offset for u/v/w (D3D10_SB_ADDRESS_OFFSET_COORD)
// This macro returns a (signed) integer, by sign extending the
// decoded 4 bit 2's complement immediate value.
#define DECODE_IMMEDIATE_D3D10_SB_ADDRESS_OFFSET(Coord,OpcodeToken1) ((((OpcodeToken1)&D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_MASK(Coord))>>(D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_SHIFT(Coord))))

// ENCODER MACRO: Store the immediate texel address offset
// for U or V or W Coord (D3D10_SB_ADDRESS_OFFSET_COORD) in an extended
// opcode token (OpcodeToken1) that has extended opcode
// type == D3D10_SB_EXTENDED_OPCODE_SAMPLE_CONTROLS (opcode type encoded separately)
// A 2's complement number is expected as input, from which the LSB 4 bits are extracted.
#define ENCODE_IMMEDIATE_D3D10_SB_ADDRESS_OFFSET(Coord,ImmediateOffset) (((ImmediateOffset)<<D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_SHIFT(Coord))&D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_MASK(Coord))

#define D3D11_SB_EXTENDED_RESOURCE_DIMENSION_MASK  0x000007C0
#define D3D11_SB_EXTENDED_RESOURCE_DIMENSION_SHIFT 6

// DECODER MACRO: Given an extended resource declaration token,
// (D3D11_SB_EXTENDED_OPCODE_RESOURCE_DIM), determine the resource dimension
// (D3D10_SB_RESOURCE_DIMENSION enum)
#define DECODE_D3D11_SB_EXTENDED_RESOURCE_DIMENSION(OpcodeTokenN) ((D3D10_SB_RESOURCE_DIMENSION)(((OpcodeTokenN)&D3D11_SB_EXTENDED_RESOURCE_DIMENSION_MASK)>>D3D11_SB_EXTENDED_RESOURCE_DIMENSION_SHIFT))

// ENCODER MACRO: Store resource dimension
// (D3D10_SB_RESOURCE_DIMENSION enum) into a
// an extended resource declaration token (D3D11_SB_EXTENDED_OPCODE_RESOURCE_DIM)
#define ENCODE_D3D11_SB_EXTENDED_RESOURCE_DIMENSION(ResourceDim) (((ResourceDim)<<D3D11_SB_EXTENDED_RESOURCE_DIMENSION_SHIFT)&D3D11_SB_EXTENDED_RESOURCE_DIMENSION_MASK)

#define D3D11_SB_EXTENDED_RESOURCE_DIMENSION_STRUCTURE_STRIDE_MASK  0x007FF800
#define D3D11_SB_EXTENDED_RESOURCE_DIMENSION_STRUCTURE_STRIDE_SHIFT 11

// DECODER MACRO: Given an extended resource declaration token for a structured buffer,
// (D3D11_SB_EXTENDED_OPCODE_RESOURCE_DIM), determine the structure stride
// (12-bit unsigned integer)
#define DECODE_D3D11_SB_EXTENDED_RESOURCE_DIMENSION_STRUCTURE_STRIDE(OpcodeTokenN) (((OpcodeTokenN)&D3D11_SB_EXTENDED_RESOURCE_DIMENSION_STRUCTURE_STRIDE_MASK)>>D3D11_SB_EXTENDED_RESOURCE_DIMENSION_STRUCTURE_STRIDE_SHIFT)

// ENCODER MACRO: Store resource dimension structure stride
// (12-bit unsigned integer) into a
// an extended resource declaration token (D3D11_SB_EXTENDED_OPCODE_RESOURCE_DIM)
#define ENCODE_D3D11_SB_EXTENDED_RESOURCE_DIMENSION_STRUCTURE_STRIDE(Stride) (((Stride)<<D3D11_SB_EXTENDED_RESOURCE_DIMENSION_STRUCTURE_STRIDE_SHIFT)&D3D11_SB_EXTENDED_RESOURCE_DIMENSION_STRUCTURE_STRIDE_MASK)

#define D3D10_SB_RESOURCE_RETURN_TYPE_MASK    0x0000000f
#define D3D10_SB_RESOURCE_RETURN_TYPE_NUMBITS 0x00000004
#define D3D11_SB_EXTENDED_RESOURCE_RETURN_TYPE_SHIFT 6

// DECODER MACRO: Get the resource return type for component (0-3) from
// an extended resource declaration token (D3D11_SB_EXTENDED_OPCODE_RESOURCE_RETURN_TYPE)
#define DECODE_D3D11_SB_EXTENDED_RESOURCE_RETURN_TYPE(OpcodeTokenN, Component) \
    ((D3D10_SB_RESOURCE_RETURN_TYPE)(((OpcodeTokenN) >> \
    (Component * D3D10_SB_RESOURCE_RETURN_TYPE_NUMBITS + D3D11_SB_EXTENDED_RESOURCE_RETURN_TYPE_SHIFT))&D3D10_SB_RESOURCE_RETURN_TYPE_MASK))

// ENCODER MACRO: Generate a resource return type for a component in an extended
// resource delcaration token (D3D11_SB_EXTENDED_OPCODE_RESOURCE_RETURN_TYPE)
#define ENCODE_D3D11_SB_EXTENDED_RESOURCE_RETURN_TYPE(ReturnType, Component) \
    (((ReturnType)&D3D10_SB_RESOURCE_RETURN_TYPE_MASK) << (Component * D3D10_SB_RESOURCE_RETURN_TYPE_NUMBITS + D3D11_SB_EXTENDED_RESOURCE_RETURN_TYPE_SHIFT))

// ----------------------------------------------------------------------------
// Custom-Data Block Format
//
// DWORD 0 (CustomDataDescTok):
// [10:00] == D3D10_SB_OPCODE_CUSTOMDATA
// [31:11] == D3D10_SB_CUSTOMDATA_CLASS
//
// DWORD 1: 
//          32-bit unsigned integer count of number
//          of DWORDs in custom-data block,
//          including DWORD 0 and DWORD 1.
//          So the minimum value is 0x00000002,
//          meaning empty custom-data.
//
// Layout of custom-data contents, for the various meta-data classes,
// not defined in this file.
//
// ----------------------------------------------------------------------------

typedef enum D3D10_SB_CUSTOMDATA_CLASS
{
    D3D10_SB_CUSTOMDATA_COMMENT = 0,
    D3D10_SB_CUSTOMDATA_DEBUGINFO,
    D3D10_SB_CUSTOMDATA_OPAQUE,
    D3D10_SB_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER,
    D3D11_SB_CUSTOMDATA_SHADER_MESSAGE,
    D3D11_SB_CUSTOMDATA_SHADER_CLIP_PLANE_CONSTANT_MAPPINGS_FOR_DX9,
} D3D10_SB_CUSTOMDATA_CLASS;

#define D3D10_SB_CUSTOMDATA_CLASS_MASK 0xfffff800
#define D3D10_SB_CUSTOMDATA_CLASS_SHIFT 11
// DECODER MACRO: Find out what class of custom-data is present.
// The contents of the custom-data block are defined
// for each class of custom-data.
#define DECODE_D3D10_SB_CUSTOMDATA_CLASS(CustomDataDescTok) ((D3D10_SB_CUSTOMDATA_CLASS)(((CustomDataDescTok)&D3D10_SB_CUSTOMDATA_CLASS_MASK)>>D3D10_SB_CUSTOMDATA_CLASS_SHIFT))
// ENCODER MACRO: Create complete CustomDataDescTok
#define ENCODE_D3D10_SB_CUSTOMDATA_CLASS(CustomDataClass) (ENCODE_D3D10_SB_OPCODE_TYPE(D3D10_SB_OPCODE_CUSTOMDATA)|(((CustomDataClass)<<D3D10_SB_CUSTOMDATA_CLASS_SHIFT)&D3D10_SB_CUSTOMDATA_CLASS_MASK))

// ----------------------------------------------------------------------------
// Instruction Operand Format (OperandToken0)
//
// [01:00] D3D10_SB_OPERAND_NUM_COMPONENTS
// [11:02] Component Selection
//         if([01:00] == D3D10_SB_OPERAND_0_COMPONENT)
//              [11:02] = Ignored, 0
//         else if([01:00] == D3D10_SB_OPERAND_1_COMPONENT
//              [11:02] = Ignored, 0
//         else if([01:00] == D3D10_SB_OPERAND_4_COMPONENT
//         {
//              [03:02] = D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE
//              if([03:02] == D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE)
//              {
//                  [07:04] = D3D10_SB_OPERAND_4_COMPONENT_MASK
//                  [11:08] = Ignored, 0
//              }
//              else if([03:02] == D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE)
//              {
//                  [11:04] = D3D10_SB_4_COMPONENT_SWIZZLE
//              }
//              else if([03:02] == D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE)
//              {
//                  [05:04] = D3D10_SB_4_COMPONENT_NAME
//                  [11:06] = Ignored, 0
//              }
//         }
//         else if([01:00] == D3D10_SB_OPERAND_N_COMPONENT)
//         {
//              Currently not defined.
//         }
// [19:12] D3D10_SB_OPERAND_TYPE
// [21:20] D3D10_SB_OPERAND_INDEX_DIMENSION:
//            Number of dimensions in the register
//            file (NOT the # of dimensions in the
//            individual register or memory
//            resource being referenced).
// [24:22] if( [21:20] >= D3D10_SB_OPERAND_INDEX_1D )
//             D3D10_SB_OPERAND_INDEX_REPRESENTATION for first operand index
//         else
//             Ignored, 0
// [27:25] if( [21:20] >= D3D10_SB_OPERAND_INDEX_2D )
//             D3D10_SB_OPERAND_INDEX_REPRESENTATION for second operand index
//         else
//             Ignored, 0
// [30:28] if( [21:20] == D3D10_SB_OPERAND_INDEX_3D )
//             D3D10_SB_OPERAND_INDEX_REPRESENTATION for third operand index
//         else
//             Ignored, 0
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.
//
// ----------------------------------------------------------------------------

// Number of components in data vector referred to by operand.
typedef enum D3D10_SB_OPERAND_NUM_COMPONENTS
{
    D3D10_SB_OPERAND_0_COMPONENT = 0,
    D3D10_SB_OPERAND_1_COMPONENT = 1,
    D3D10_SB_OPERAND_4_COMPONENT = 2,
    D3D10_SB_OPERAND_N_COMPONENT = 3 // unused for now
} D3D10_SB_OPERAND_NUM_COMPONENTS;
#define D3D10_SB_OPERAND_NUM_COMPONENTS_MASK 0x00000003

// DECODER MACRO: Extract from OperandToken0 how many components
// the data vector referred to by the operand contains.
// (D3D10_SB_OPERAND_NUM_COMPONENTS enum)
#define DECODE_D3D10_SB_OPERAND_NUM_COMPONENTS(OperandToken0) ((D3D10_SB_OPERAND_NUM_COMPONENTS)((OperandToken0)&D3D10_SB_OPERAND_NUM_COMPONENTS_MASK))

// ENCODER MACRO: Define in OperandToken0 how many components
// the data vector referred to by the operand contains.
// (D3D10_SB_OPERAND_NUM_COMPONENTS enum).
#define ENCODE_D3D10_SB_OPERAND_NUM_COMPONENTS(NumComp) ((NumComp)&D3D10_SB_OPERAND_NUM_COMPONENTS_MASK)

typedef enum D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE
{
    D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE    = 0,  // mask 4 components
    D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE = 1,  // swizzle 4 components
    D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE = 2, // select 1 of 4 components
} D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE;

#define D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE_MASK  0x0000000c
#define D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE_SHIFT 2

// DECODER MACRO: For an operand representing 4component data,
// extract from OperandToken0 the method for selecting data from
// the 4 components (D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE).
#define DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(OperandToken0) ((D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE)(((OperandToken0)&D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE_MASK)>>D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE_SHIFT))

// ENCODER MACRO: For an operand representing 4component data,
// encode in OperandToken0 a value from D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE
#define ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(SelectionMode) (((SelectionMode)<<D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE_SHIFT)&D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE_MASK)

typedef enum D3D10_SB_4_COMPONENT_NAME
{
    D3D10_SB_4_COMPONENT_X = 0,
    D3D10_SB_4_COMPONENT_Y = 1,
    D3D10_SB_4_COMPONENT_Z = 2,
    D3D10_SB_4_COMPONENT_W = 3,
    D3D10_SB_4_COMPONENT_R = 0,
    D3D10_SB_4_COMPONENT_G = 1,
    D3D10_SB_4_COMPONENT_B = 2,
    D3D10_SB_4_COMPONENT_A = 3
} D3D10_SB_4_COMPONENT_NAME;
#define D3D10_SB_4_COMPONENT_NAME_MASK 3

// MACROS FOR USE IN D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE:

#define D3D10_SB_OPERAND_4_COMPONENT_MASK_MASK   0x000000f0
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT  4
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_X      0x00000010
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_Y      0x00000020
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_Z      0x00000040
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_W      0x00000080
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_R      D3D10_SB_OPERAND_4_COMPONENT_MASK_X
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_G      D3D10_SB_OPERAND_4_COMPONENT_MASK_Y
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_B      D3D10_SB_OPERAND_4_COMPONENT_MASK_Z
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_A      D3D10_SB_OPERAND_4_COMPONENT_MASK_W
#define D3D10_SB_OPERAND_4_COMPONENT_MASK_ALL    D3D10_SB_OPERAND_4_COMPONENT_MASK_MASK

// DECODER MACRO: When 4 component selection mode is
// D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE, this macro
// extracts from OperandToken0 the 4 component (xyzw) mask,
// as a field of D3D10_SB_OPERAND_4_COMPONENT_MASK_[X|Y|Z|W] flags.
// Alternatively, the D3D10_SB_OPERAND_4_COMPONENT_MASK_[X|Y|Z|W] masks
// can be tested on OperandToken0 directly, without this macro.
#define DECODE_D3D10_SB_OPERAND_4_COMPONENT_MASK(OperandToken0) ((OperandToken0)&D3D10_SB_OPERAND_4_COMPONENT_MASK_MASK)

// ENCODER MACRO: Given a set of
// D3D10_SB_OPERAND_4_COMPONENT_MASK_[X|Y|Z|W] values
// or'd together, encode them in OperandToken0.
#define ENCODE_D3D10_SB_OPERAND_4_COMPONENT_MASK(ComponentMask) ((ComponentMask)&D3D10_SB_OPERAND_4_COMPONENT_MASK_MASK)

// ENCODER/DECODER MACRO: Given a D3D10_SB_4_COMPONENT_NAME,
// generate the 4-component mask for it.
// This can be used in loops that build masks or read masks.
// Alternatively, the D3D10_SB_OPERAND_4_COMPONENT_MASK_[X|Y|Z|W] masks
// can be used directly, without this macro.
#define D3D10_SB_OPERAND_4_COMPONENT_MASK(ComponentName) ((1<<(D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT+ComponentName))&D3D10_SB_OPERAND_4_COMPONENT_MASK_MASK)

// MACROS FOR USE IN D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE:

#define D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MASK 0x00000ff0
#define D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SHIFT 4

// DECODER MACRO: When 4 component selection mode is
// D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE, this macro
// extracts from OperandToken0 the 4 component swizzle,
// as a field of D3D10_SB_OPERAND_4_COMPONENT_MASK_[X|Y|Z|W] flags.
#define DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE(OperandToken0) ((OperandToken0)&D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MASK)

// DECODER MACRO: Pass a D3D10_SB_4_COMPONENT_NAME as "DestComp" in following
// macro to extract, from OperandToken0 or from a decoded swizzle,
// the swizzle source component (D3D10_SB_4_COMPONENT_NAME enum):
#define DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(OperandToken0,DestComp) ((D3D10_SB_4_COMPONENT_NAME)(((OperandToken0)>>(D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SHIFT+2*((DestComp)&D3D10_SB_4_COMPONENT_NAME_MASK)))&D3D10_SB_4_COMPONENT_NAME_MASK))

// ENCODER MACRO: Generate a 4 component swizzle given
// 4 D3D10_SB_4_COMPONENT_NAME source values for dest
// components x, y, z, w respectively.
#define ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE(XSrc,YSrc,ZSrc,WSrc) ((((XSrc)&D3D10_SB_4_COMPONENT_NAME_MASK)|     \
                                                                     (((YSrc)&D3D10_SB_4_COMPONENT_NAME_MASK)<<2)| \
                                                                     (((ZSrc)&D3D10_SB_4_COMPONENT_NAME_MASK)<<4)| \
                                                                     (((WSrc)&D3D10_SB_4_COMPONENT_NAME_MASK)<<6)  \
                                                                      )<<D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SHIFT)

// ENCODER/DECODER MACROS: Various common swizzle patterns
// (noswizzle and replicate of each channels)
#define D3D10_SB_OPERAND_4_COMPONENT_NOSWIZZLE   ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE(D3D10_SB_4_COMPONENT_X,\
                                                                                   D3D10_SB_4_COMPONENT_Y,\
                                                                                   D3D10_SB_4_COMPONENT_Z,\
                                                                                   D3D10_SB_4_COMPONENT_W)

#define D3D10_SB_OPERAND_4_COMPONENT_REPLICATEX  ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE(D3D10_SB_4_COMPONENT_X,\
                                                                                   D3D10_SB_4_COMPONENT_X,\
                                                                                   D3D10_SB_4_COMPONENT_X,\
                                                                                   D3D10_SB_4_COMPONENT_X)

#define D3D10_SB_OPERAND_4_COMPONENT_REPLICATEY  ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE(D3D10_SB_4_COMPONENT_Y,\
                                                                                   D3D10_SB_4_COMPONENT_Y,\
                                                                                   D3D10_SB_4_COMPONENT_Y,\
                                                                                   D3D10_SB_4_COMPONENT_Y)

#define D3D10_SB_OPERAND_4_COMPONENT_REPLICATEZ  ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE(D3D10_SB_4_COMPONENT_Z,\
                                                                                   D3D10_SB_4_COMPONENT_Z,\
                                                                                   D3D10_SB_4_COMPONENT_Z,\
                                                                                   D3D10_SB_4_COMPONENT_Z)

#define D3D10_SB_OPERAND_4_COMPONENT_REPLICATEW  ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE(D3D10_SB_4_COMPONENT_W,\
                                                                                   D3D10_SB_4_COMPONENT_W,\
                                                                                   D3D10_SB_4_COMPONENT_W,\
                                                                                   D3D10_SB_4_COMPONENT_W)

#define D3D10_SB_OPERAND_4_COMPONENT_REPLICATERED    D3D10_SB_OPERAND_4_COMPONENT_REPLICATEX
#define D3D10_SB_OPERAND_4_COMPONENT_REPLICATEGREEN  D3D10_SB_OPERAND_4_COMPONENT_REPLICATEY
#define D3D10_SB_OPERAND_4_COMPONENT_REPLICATEBLUE   D3D10_SB_OPERAND_4_COMPONENT_REPLICATEZ
#define D3D10_SB_OPERAND_4_COMPONENT_REPLICATEALPHA  D3D10_SB_OPERAND_4_COMPONENT_REPLICATEW

// MACROS FOR USE IN D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE:
#define D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MASK   0x00000030
#define D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_SHIFT  4

// DECODER MACRO: When 4 component selection mode is
// D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE, this macro
// extracts from OperandToken0 a D3D10_SB_4_COMPONENT_NAME
// which picks one of the 4 components.
#define DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECT_1(OperandToken0) ((D3D10_SB_4_COMPONENT_NAME)(((OperandToken0)&D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MASK)>>D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_SHIFT))

// ENCODER MACRO: Given a D3D10_SB_4_COMPONENT_NAME selecting
// a single component for D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE,
// encode it into OperandToken0
#define ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SELECT_1(SelectedComp) (((SelectedComp)<<D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_SHIFT)&D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MASK)

// MACROS FOR DETERMINING OPERAND TYPE:

typedef enum D3D10_SB_OPERAND_TYPE
{
    D3D10_SB_OPERAND_TYPE_TEMP           = 0,  // Temporary Register File
    D3D10_SB_OPERAND_TYPE_INPUT          = 1,  // General Input Register File
    D3D10_SB_OPERAND_TYPE_OUTPUT         = 2,  // General Output Register File
    D3D10_SB_OPERAND_TYPE_INDEXABLE_TEMP = 3,  // Temporary Register File (indexable)
    D3D10_SB_OPERAND_TYPE_IMMEDIATE32    = 4,  // 32bit/component immediate value(s)
                                          // If for example, operand token bits
                                          // [01:00]==D3D10_SB_OPERAND_4_COMPONENT,
                                          // this means that the operand type:
                                          // D3D10_SB_OPERAND_TYPE_IMMEDIATE32
                                          // results in 4 additional 32bit
                                          // DWORDS present for the operand.
    D3D10_SB_OPERAND_TYPE_IMMEDIATE64    = 5,  // 64bit/comp.imm.val(s)HI:LO
    D3D10_SB_OPERAND_TYPE_SAMPLER        = 6,  // Reference to sampler state
    D3D10_SB_OPERAND_TYPE_RESOURCE       = 7,  // Reference to memory resource (e.g. texture)
    D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER= 8,  // Reference to constant buffer
    D3D10_SB_OPERAND_TYPE_IMMEDIATE_CONSTANT_BUFFER= 9,  // Reference to immediate constant buffer
    D3D10_SB_OPERAND_TYPE_LABEL          = 10, // Label
    D3D10_SB_OPERAND_TYPE_INPUT_PRIMITIVEID = 11, // Input primitive ID
    D3D10_SB_OPERAND_TYPE_OUTPUT_DEPTH   = 12, // Output Depth
    D3D10_SB_OPERAND_TYPE_NULL           = 13, // Null register, used to discard results of operations
                                               // Below Are operands new in DX 10.1
    D3D10_SB_OPERAND_TYPE_RASTERIZER     = 14, // DX10.1 Rasterizer register, used to denote the depth/stencil and render target resources
    D3D10_SB_OPERAND_TYPE_OUTPUT_COVERAGE_MASK = 15, // DX10.1 PS output MSAA coverage mask (scalar)
                                               // Below Are operands new in DX 11
    D3D11_SB_OPERAND_TYPE_STREAM         = 16, // Reference to GS stream output resource
    D3D11_SB_OPERAND_TYPE_FUNCTION_BODY  = 17, // Reference to a function definition
    D3D11_SB_OPERAND_TYPE_FUNCTION_TABLE = 18, // Reference to a set of functions used by a class
    D3D11_SB_OPERAND_TYPE_INTERFACE      = 19, // Reference to an interface
    D3D11_SB_OPERAND_TYPE_FUNCTION_INPUT = 20, // Reference to an input parameter to a function
    D3D11_SB_OPERAND_TYPE_FUNCTION_OUTPUT = 21, // Reference to an output parameter to a function
    D3D11_SB_OPERAND_TYPE_OUTPUT_CONTROL_POINT_ID = 22, // HS Control Point phase input saying which output control point ID this is
    D3D11_SB_OPERAND_TYPE_INPUT_FORK_INSTANCE_ID = 23, // HS Fork Phase input instance ID
    D3D11_SB_OPERAND_TYPE_INPUT_JOIN_INSTANCE_ID = 24, // HS Join Phase input instance ID
    D3D11_SB_OPERAND_TYPE_INPUT_CONTROL_POINT = 25, // HS Fork+Join, DS phase input control points (array of them)
    D3D11_SB_OPERAND_TYPE_OUTPUT_CONTROL_POINT = 26, // HS Fork+Join phase output control points (array of them)
    D3D11_SB_OPERAND_TYPE_INPUT_PATCH_CONSTANT = 27, // DS+HSJoin Input Patch Constants (array of them)
    D3D11_SB_OPERAND_TYPE_INPUT_DOMAIN_POINT = 28, // DS Input Domain point
    D3D11_SB_OPERAND_TYPE_THIS_POINTER       = 29, // Reference to an interface this pointer
    D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW = 30, // Reference to UAV u#
    D3D11_SB_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY = 31, // Reference to Thread Group Shared Memory g#
    D3D11_SB_OPERAND_TYPE_INPUT_THREAD_ID = 32, // Compute Shader Thread ID
    D3D11_SB_OPERAND_TYPE_INPUT_THREAD_GROUP_ID = 33, // Compute Shader Thread Group ID
    D3D11_SB_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP = 34, // Compute Shader Thread ID In Thread Group
    D3D11_SB_OPERAND_TYPE_INPUT_COVERAGE_MASK = 35, // Pixel shader coverage mask input
    D3D11_SB_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP_FLATTENED = 36, // Compute Shader Thread ID In Group Flattened to a 1D value.
    D3D11_SB_OPERAND_TYPE_INPUT_GS_INSTANCE_ID = 37, // Input GS instance ID
    D3D11_SB_OPERAND_TYPE_OUTPUT_DEPTH_GREATER_EQUAL = 38, // Output Depth, forced to be greater than or equal than current depth
    D3D11_SB_OPERAND_TYPE_OUTPUT_DEPTH_LESS_EQUAL    = 39, // Output Depth, forced to be less than or equal to current depth
    D3D11_SB_OPERAND_TYPE_CYCLE_COUNTER = 40, // Cycle counter
    D3D11_SB_OPERAND_TYPE_OUTPUT_STENCIL_REF = 41, // DX11 PS output stencil reference (scalar)
    D3D11_SB_OPERAND_TYPE_INNER_COVERAGE = 42, // DX11 PS input inner coverage (scalar)
} D3D10_SB_OPERAND_TYPE;

#define D3D10_SB_OPERAND_TYPE_MASK   0x000ff000
#define D3D10_SB_OPERAND_TYPE_SHIFT  12

// DECODER MACRO: Determine operand type from OperandToken0.
#define DECODE_D3D10_SB_OPERAND_TYPE(OperandToken0) ((D3D10_SB_OPERAND_TYPE)(((OperandToken0)&D3D10_SB_OPERAND_TYPE_MASK)>>D3D10_SB_OPERAND_TYPE_SHIFT))

// ENCODER MACRO: Store operand type in OperandToken0.
#define ENCODE_D3D10_SB_OPERAND_TYPE(OperandType) (((OperandType)<<D3D10_SB_OPERAND_TYPE_SHIFT)&D3D10_SB_OPERAND_TYPE_MASK)

typedef enum D3D10_SB_OPERAND_INDEX_DIMENSION
{
    D3D10_SB_OPERAND_INDEX_0D = 0, // e.g. Position
    D3D10_SB_OPERAND_INDEX_1D = 1, // Most common.  e.g. Temp registers.
    D3D10_SB_OPERAND_INDEX_2D = 2, // e.g. Geometry Program Input registers.
    D3D10_SB_OPERAND_INDEX_3D = 3, // 3D rarely if ever used.
} D3D10_SB_OPERAND_INDEX_DIMENSION;
#define D3D10_SB_OPERAND_INDEX_DIMENSION_MASK  0x00300000
#define D3D10_SB_OPERAND_INDEX_DIMENSION_SHIFT 20

// DECODER MACRO: Determine operand index dimension from OperandToken0.
#define DECODE_D3D10_SB_OPERAND_INDEX_DIMENSION(OperandToken0) ((D3D10_SB_OPERAND_INDEX_DIMENSION)(((OperandToken0)&D3D10_SB_OPERAND_INDEX_DIMENSION_MASK)>>D3D10_SB_OPERAND_INDEX_DIMENSION_SHIFT))

// ENCODER MACRO: Store operand index dimension
// (D3D10_SB_OPERAND_INDEX_DIMENSION enum) in OperandToken0.
#define ENCODE_D3D10_SB_OPERAND_INDEX_DIMENSION(OperandIndexDim) (((OperandIndexDim)<<D3D10_SB_OPERAND_INDEX_DIMENSION_SHIFT)&D3D10_SB_OPERAND_INDEX_DIMENSION_MASK)

typedef enum D3D10_SB_OPERAND_INDEX_REPRESENTATION
{
    D3D10_SB_OPERAND_INDEX_IMMEDIATE32               = 0, // Extra DWORD
    D3D10_SB_OPERAND_INDEX_IMMEDIATE64               = 1, // 2 Extra DWORDs
                                                     //   (HI32:LO32)
    D3D10_SB_OPERAND_INDEX_RELATIVE                  = 2, // Extra operand
    D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE = 3, // Extra DWORD followed by
                                                     //   extra operand
    D3D10_SB_OPERAND_INDEX_IMMEDIATE64_PLUS_RELATIVE = 4, // 2 Extra DWORDS
                                                     //   (HI32:LO32) followed
                                                     //   by extra operand
} D3D10_SB_OPERAND_INDEX_REPRESENTATION;
#define D3D10_SB_OPERAND_INDEX_REPRESENTATION_SHIFT(Dim) (22+3*((Dim)&3))
#define D3D10_SB_OPERAND_INDEX_REPRESENTATION_MASK(Dim) (0x3<<D3D10_SB_OPERAND_INDEX_REPRESENTATION_SHIFT(Dim))

// DECODER MACRO: Determine from OperandToken0 what representation
// an operand index is provided as (D3D10_SB_OPERAND_INDEX_REPRESENTATION enum),
// for index dimension [0], [1] or [2], depending on D3D10_SB_OPERAND_INDEX_DIMENSION.
#define DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(Dim,OperandToken0) ((D3D10_SB_OPERAND_INDEX_REPRESENTATION)(((OperandToken0)&D3D10_SB_OPERAND_INDEX_REPRESENTATION_MASK(Dim))>>D3D10_SB_OPERAND_INDEX_REPRESENTATION_SHIFT(Dim)))

// ENCODER MACRO: Store in OperandToken0 what representation
// an operand index is provided as (D3D10_SB_OPERAND_INDEX_REPRESENTATION enum),
// for index dimension [0], [1] or [2], depending on D3D10_SB_OPERAND_INDEX_DIMENSION.
#define ENCODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(Dim,IndexRepresentation) (((IndexRepresentation)<<D3D10_SB_OPERAND_INDEX_REPRESENTATION_SHIFT(Dim))&D3D10_SB_OPERAND_INDEX_REPRESENTATION_MASK(Dim))

#define D3D10_SB_OPERAND_EXTENDED_MASK  0x80000000
#define D3D10_SB_OPERAND_EXTENDED_SHIFT 31

// DECODER MACRO: Determine if the operand is extended
// by an additional opcode token.
#define DECODE_IS_D3D10_SB_OPERAND_EXTENDED(OperandToken0) (((OperandToken0)&D3D10_SB_OPERAND_EXTENDED_MASK)>>D3D10_SB_OPERAND_EXTENDED_SHIFT)

// ENCODER MACRO: Store in OperandToken0 whether the operand is extended
// by an additional operand token.
#define ENCODE_D3D10_SB_OPERAND_EXTENDED(bExtended) (((bExtended)!=0)?D3D10_SB_OPERAND_EXTENDED_MASK:0)

// ----------------------------------------------------------------------------
// Extended Instruction Operand Format (OperandToken1)
//
// If bit31 of an operand token is set, the
// operand has additional data in a second DWORD
// directly following OperandToken0.  Other tokens
// expected for the operand, such as immmediate
// values or relative address operands (full
// operands in themselves) always follow
// OperandToken0 AND OperandToken1..n (extended
// operand tokens, if present).
//
// [05:00] D3D10_SB_EXTENDED_OPERAND_TYPE
// [16:06] if([05:00] == D3D10_SB_EXTENDED_OPERAND_MODIFIER)
//         {
//              [13:06] D3D10_SB_OPERAND_MODIFIER
//              [16:14] Min Precision: D3D11_SB_OPERAND_MIN_PRECISION
//              [17:17] Non-uniform: D3D12_SB_OPERAND_NON_UNIFORM
//         }
//         else
//         {
//              [17:06] Ignored, 0.
//         }
// [30:18] Ignored, 0.
// [31]    0 normally. 1 if second order extended operand definition,
//         meaning next DWORD contains yet ANOTHER extended operand
//         description. Currently no second order extensions defined.
//         This would be useful if a particular extended operand does
//         not have enough space to store the required information in
//         a single token and so is extended further.
//
// ----------------------------------------------------------------------------

typedef enum D3D10_SB_EXTENDED_OPERAND_TYPE
{
    D3D10_SB_EXTENDED_OPERAND_EMPTY            = 0, // Might be used if this
                                               // enum is full and
                                               // further extended opcode
                                               // is needed.
    D3D10_SB_EXTENDED_OPERAND_MODIFIER         = 1,
} D3D10_SB_EXTENDED_OPERAND_TYPE;
#define D3D10_SB_EXTENDED_OPERAND_TYPE_MASK 0x0000003f

// DECODER MACRO: Given an extended operand
// token (OperandToken1), figure out what type
// of token it is (from D3D10_SB_EXTENDED_OPERAND_TYPE enum)
// to be able to interpret the rest of the token's contents.
#define DECODE_D3D10_SB_EXTENDED_OPERAND_TYPE(OperandToken1) ((D3D10_SB_EXTENDED_OPERAND_TYPE)((OperandToken1)&D3D10_SB_EXTENDED_OPERAND_TYPE_MASK))

// ENCODER MACRO: Store extended operand token
// type in OperandToken1.
#define ENCODE_D3D10_SB_EXTENDED_OPERAND_TYPE(ExtOperandType) ((ExtOperandType)&D3D10_SB_EXTENDED_OPERAND_TYPE_MASK)

typedef enum D3D10_SB_OPERAND_MODIFIER
{
    D3D10_SB_OPERAND_MODIFIER_NONE     = 0, // Nop.  This is the implied
                                             // default if the extended
                                             // operand is not present for
                                             // an operand for which source
                                             // modifiers are meaningful
    D3D10_SB_OPERAND_MODIFIER_NEG      = 1, // Negate
    D3D10_SB_OPERAND_MODIFIER_ABS      = 2, // Absolute value, abs()
    D3D10_SB_OPERAND_MODIFIER_ABSNEG   = 3, // -abs()
} D3D10_SB_OPERAND_MODIFIER;
#define D3D10_SB_OPERAND_MODIFIER_MASK  0x00003fc0
#define D3D10_SB_OPERAND_MODIFIER_SHIFT 6

// DECODER MACRO: Given a D3D10_SB_EXTENDED_OPERAND_MODIFIER
// extended token (OperandToken1), determine the source modifier
// (D3D10_SB_OPERAND_MODIFIER enum)
#define DECODE_D3D10_SB_OPERAND_MODIFIER(OperandToken1) ((D3D10_SB_OPERAND_MODIFIER)(((OperandToken1)&D3D10_SB_OPERAND_MODIFIER_MASK)>>D3D10_SB_OPERAND_MODIFIER_SHIFT))

// ENCODER MACRO: Generate a complete source modifier extended token
// (OperandToken1), given D3D10_SB_OPERAND_MODIFIER enum (the
// ext. operand type is also set to D3D10_SB_EXTENDED_OPERAND_MODIFIER).
#define ENCODE_D3D10_SB_EXTENDED_OPERAND_MODIFIER(SourceMod)  ((((SourceMod)<<D3D10_SB_OPERAND_MODIFIER_SHIFT)&D3D10_SB_OPERAND_MODIFIER_MASK)| \
                                                                ENCODE_D3D10_SB_EXTENDED_OPERAND_TYPE(D3D10_SB_EXTENDED_OPERAND_MODIFIER) | \
                                                                ENCODE_D3D10_SB_OPERAND_DOUBLE_EXTENDED(0))

// Min precision specifier for source/dest operands.  This 
// fits in the extended operand token field. Implementations are free to 
// execute at higher precision than the min - details spec'ed elsewhere.
// This is part of the opcode specific control range.
typedef enum D3D11_SB_OPERAND_MIN_PRECISION
{
    D3D11_SB_OPERAND_MIN_PRECISION_DEFAULT    = 0, // Default precision 
                                                       // for the shader model
    D3D11_SB_OPERAND_MIN_PRECISION_FLOAT_16   = 1, // Min 16 bit/component float
    D3D11_SB_OPERAND_MIN_PRECISION_FLOAT_2_8  = 2, // Min 10(2.8)bit/comp. float
    D3D11_SB_OPERAND_MIN_PRECISION_SINT_16    = 4, // Min 16 bit/comp. signed integer
    D3D11_SB_OPERAND_MIN_PRECISION_UINT_16    = 5, // Min 16 bit/comp. unsigned integer
} D3D11_SB_OPERAND_MIN_PRECISION;
#define D3D11_SB_OPERAND_MIN_PRECISION_MASK  0x0001C000
#define D3D11_SB_OPERAND_MIN_PRECISION_SHIFT 14

// DECODER MACRO: For an OperandToken1 that can specify
// a minimum precision for execution, find out what it is.
#define DECODE_D3D11_SB_OPERAND_MIN_PRECISION(OperandToken1) ((D3D11_SB_OPERAND_MIN_PRECISION)(((OperandToken1)& D3D11_SB_OPERAND_MIN_PRECISION_MASK)>> D3D11_SB_OPERAND_MIN_PRECISION_SHIFT))

// ENCODER MACRO: Encode minimum precision for execution
// into the extended operand token, OperandToken1
#define ENCODE_D3D11_SB_OPERAND_MIN_PRECISION(MinPrecision) (((MinPrecision)<< D3D11_SB_OPERAND_MIN_PRECISION_SHIFT)& D3D11_SB_OPERAND_MIN_PRECISION_MASK)


// Non-uniform extended operand modifier.
#define D3D12_SB_OPERAND_NON_UNIFORM_MASK  0x00020000
#define D3D12_SB_OPERAND_NON_UNIFORM_SHIFT 17

// DECODER MACRO: For an OperandToken1 that can specify a non-uniform operand
#define DECODE_D3D12_SB_OPERAND_NON_UNIFORM(OperandToken1) (((OperandToken1)& D3D12_SB_OPERAND_NON_UNIFORM_MASK)>> D3D12_SB_OPERAND_NON_UNIFORM_SHIFT)

// ENCODER MACRO: Encode non-uniform state into the extended operand token, OperandToken1
#define ENCODE_D3D12_SB_OPERAND_NON_UNIFORM(NonUniform) (((NonUniform)<< D3D12_SB_OPERAND_NON_UNIFORM_SHIFT)& D3D12_SB_OPERAND_NON_UNIFORM_MASK)


#define D3D10_SB_OPERAND_DOUBLE_EXTENDED_MASK  0x80000000
#define D3D10_SB_OPERAND_DOUBLE_EXTENDED_SHIFT 31
// DECODER MACRO: Determine if an extended operand token
// (OperandToken1) is further extended by yet another token
// (OperandToken2).  Currently there are no secondary
// extended operand tokens.
#define DECODE_IS_D3D10_SB_OPERAND_DOUBLE_EXTENDED(OperandToken1) (((OperandToken1)&D3D10_SB_OPERAND_DOUBLE_EXTENDED_MASK)>>D3D10_SB_OPERAND_DOUBLE_EXTENDED_SHIFT)

// ENCODER MACRO: Store in OperandToken1 whether the operand is extended
// by an additional operand token.  Currently there are no secondary
// extended operand tokens.
#define ENCODE_D3D10_SB_OPERAND_DOUBLE_EXTENDED(bExtended) (((bExtended)!=0)?D3D10_SB_OPERAND_DOUBLE_EXTENDED_MASK:0)

// ----------------------------------------------------------------------------
// Name Token (NameToken) (used in declaration statements)
//
// [15:00] D3D10_SB_NAME enumeration
// [31:16] Reserved, 0
//
// ----------------------------------------------------------------------------
#define D3D10_SB_NAME_MASK  0x0000ffff

// DECODER MACRO: Get the name from NameToken
#define DECODE_D3D10_SB_NAME(NameToken) ((D3D10_SB_NAME)((NameToken)&D3D10_SB_NAME_MASK))

// ENCODER MACRO: Generate a complete NameToken given a D3D10_SB_NAME
#define ENCODE_D3D10_SB_NAME(Name) ((Name)&D3D10_SB_NAME_MASK)

//---------------------------------------------------------------------
// Declaration Statements
//
// Declarations start with a standard opcode token,
// having opcode type being D3D10_SB_OPCODE_DCL*.
// Each particular declaration type has custom
// operand token(s), described below.
//---------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Global Flags Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS
// [11:11] Refactoring allowed if bit set.
// [12:12] Enable double precision float ops.
// [13:13] Force early depth-stencil test.
// [14:14] Enable RAW and structured buffers in non-CS 4.x shaders.
// [15:15] Skip optimizations of shader IL when translating to native code
// [16:16] Enable minimum-precision data types
// [17:17] Enable 11.1 double-precision floating-point instruction extensions
// [18:18] Enable 11.1 non-double instruction extensions
// [23:19] Reserved for future flags.
// [30:24] Instruction length in DWORDs including the opcode token. == 1
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by no operands.
//
// ----------------------------------------------------------------------------
#define D3D10_SB_GLOBAL_FLAG_REFACTORING_ALLOWED               (1<<11)
#define D3D11_SB_GLOBAL_FLAG_ENABLE_DOUBLE_PRECISION_FLOAT_OPS (1<<12)
#define D3D11_SB_GLOBAL_FLAG_FORCE_EARLY_DEPTH_STENCIL         (1<<13)
#define D3D11_SB_GLOBAL_FLAG_ENABLE_RAW_AND_STRUCTURED_BUFFERS (1<<14)
#define D3D11_1_SB_GLOBAL_FLAG_SKIP_OPTIMIZATION               (1<<15)
#define D3D11_1_SB_GLOBAL_FLAG_ENABLE_MINIMUM_PRECISION        (1<<16)
#define D3D11_1_SB_GLOBAL_FLAG_ENABLE_DOUBLE_EXTENSIONS        (1<<17)
#define D3D11_1_SB_GLOBAL_FLAG_ENABLE_SHADER_EXTENSIONS        (1<<18)
#define D3D12_SB_GLOBAL_FLAG_ALL_RESOURCES_BOUND               (1<<19)

#define D3D10_SB_GLOBAL_FLAGS_MASK  0x00fff800

// DECODER MACRO: Get global flags
#define DECODE_D3D10_SB_GLOBAL_FLAGS(OpcodeToken0) ((OpcodeToken0)&D3D10_SB_GLOBAL_FLAGS_MASK)

// ENCODER MACRO: Encode global flags
#define ENCODE_D3D10_SB_GLOBAL_FLAGS(Flags) ((Flags)&D3D10_SB_GLOBAL_FLAGS_MASK)

// ----------------------------------------------------------------------------
// Resource Declaration (non multisampled)
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_RESOURCE
// [15:11] D3D10_SB_RESOURCE_DIMENSION
// [23:16] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands on Shader Models 4.0 through 5.0:
// (1) an operand, starting with OperandToken0, defining which
//     t# register (D3D10_SB_OPERAND_TYPE_RESOURCE) is being declared.
// (2) a Resource Return Type token (ResourceReturnTypeToken)
//
// OpcodeToken0 is followed by 3 operands on Shader Model 5.1 and later:
// (1) an operand, starting with OperandToken0, defining which
//     t# register (D3D10_SB_OPERAND_TYPE_RESOURCE) is being declared.
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (t<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of resources in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the t# is used in shader instructions, where the register
//     must be D3D10_SB_OPERAND_INDEX_DIMENSION_2D, and the meaning of the index 
//     dimensions are as follows: (t<id>[<idx>]):
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of resource within space (may be dynamically indexed)
// (2) a Resource Return Type token (ResourceReturnTypeToken)
// (3) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------
#define D3D10_SB_RESOURCE_DIMENSION_MASK  0x0000F800
#define D3D10_SB_RESOURCE_DIMENSION_SHIFT 11

// DECODER MACRO: Given a resource declaration token,
// (OpcodeToken0), determine the resource dimension
// (D3D10_SB_RESOURCE_DIMENSION enum)
#define DECODE_D3D10_SB_RESOURCE_DIMENSION(OpcodeToken0) ((D3D10_SB_RESOURCE_DIMENSION)(((OpcodeToken0)&D3D10_SB_RESOURCE_DIMENSION_MASK)>>D3D10_SB_RESOURCE_DIMENSION_SHIFT))

// ENCODER MACRO: Store resource dimension
// (D3D10_SB_RESOURCE_DIMENSION enum) into a
// a resource declaration token (OpcodeToken0)
#define ENCODE_D3D10_SB_RESOURCE_DIMENSION(ResourceDim) (((ResourceDim)<<D3D10_SB_RESOURCE_DIMENSION_SHIFT)&D3D10_SB_RESOURCE_DIMENSION_MASK)

// ----------------------------------------------------------------------------
// Resource Declaration (multisampled)
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_RESOURCE (same opcode as non-multisampled case)
// [15:11] D3D10_SB_RESOURCE_DIMENSION (must be TEXTURE2DMS or TEXTURE2DMSARRAY)
// [22:16] Sample count 1...127.  0 is currently disallowed, though
//         in future versions 0 could mean "configurable" sample count
// [23:23] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands on Shader Models 4.0 through 5.0:
// (1) an operand, starting with OperandToken0, defining which
//     t# register (D3D10_SB_OPERAND_TYPE_RESOURCE) is being declared.
// (2) a Resource Return Type token (ResourceReturnTypeToken)
//
// OpcodeToken0 is followed by 3 operands on Shader Model 5.1 and later:
// (1) an operand, starting with OperandToken0, defining which
//     t# register (D3D10_SB_OPERAND_TYPE_RESOURCE) is being declared.
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (t<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of resources in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the t# is used in shader instructions, where the register
//     must be D3D10_SB_OPERAND_INDEX_DIMENSION_2D, and the meaning of the index 
//     dimensions are as follows: (t<id>[<idx>]):
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of resource within space (may be dynamically indexed)
// (2) a Resource Return Type token (ResourceReturnTypeToken)
// (3) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------

// use same macro for encoding/decoding resource dimension aas the non-msaa declaration

#define D3D10_SB_RESOURCE_SAMPLE_COUNT_MASK  0x07F0000
#define D3D10_SB_RESOURCE_SAMPLE_COUNT_SHIFT 16

// DECODER MACRO: Given a resource declaration token,
// (OpcodeToken0), determine the resource sample count (1..127)
#define DECODE_D3D10_SB_RESOURCE_SAMPLE_COUNT(OpcodeToken0) ((UINT)(((OpcodeToken0)&D3D10_SB_RESOURCE_SAMPLE_COUNT_MASK)>>D3D10_SB_RESOURCE_SAMPLE_COUNT_SHIFT))

// ENCODER MACRO: Store resource sample count up to 127 into a
// a resource declaration token (OpcodeToken0)
#define ENCODE_D3D10_SB_RESOURCE_SAMPLE_COUNT(SampleCount) (((SampleCount > 127 ? 127 : SampleCount)<<D3D10_SB_RESOURCE_SAMPLE_COUNT_SHIFT)&D3D10_SB_RESOURCE_SAMPLE_COUNT_MASK)

// ----------------------------------------------------------------------------
// Resource Return Type Token (ResourceReturnTypeToken) (used in resource
// declaration statements)
//
// [03:00] D3D10_SB_RESOURCE_RETURN_TYPE for component X
// [07:04] D3D10_SB_RESOURCE_RETURN_TYPE for component Y
// [11:08] D3D10_SB_RESOURCE_RETURN_TYPE for component Z
// [15:12] D3D10_SB_RESOURCE_RETURN_TYPE for component W
// [31:16] Reserved, 0
//
// ----------------------------------------------------------------------------
// DECODER MACRO: Get the resource return type for component (0-3) from
// ResourceReturnTypeToken
#define DECODE_D3D10_SB_RESOURCE_RETURN_TYPE(ResourceReturnTypeToken, Component) \
    ((D3D10_SB_RESOURCE_RETURN_TYPE)(((ResourceReturnTypeToken) >> \
    (Component * D3D10_SB_RESOURCE_RETURN_TYPE_NUMBITS))&D3D10_SB_RESOURCE_RETURN_TYPE_MASK))

// ENCODER MACRO: Generate a resource return type for a component
#define ENCODE_D3D10_SB_RESOURCE_RETURN_TYPE(ReturnType, Component) \
    (((ReturnType)&D3D10_SB_RESOURCE_RETURN_TYPE_MASK) << (Component * D3D10_SB_RESOURCE_RETURN_TYPE_NUMBITS))

// ----------------------------------------------------------------------------
// Sampler Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_SAMPLER
// [14:11] D3D10_SB_SAMPLER_MODE
// [23:15] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 1 operand on Shader Models 4.0 through 5.0:
// (1) Operand starting with OperandToken0, defining which sampler
//     (D3D10_SB_OPERAND_TYPE_SAMPLER) register # is being declared.
//
// OpcodeToken0 is followed by 2 operands on Shader Model 5.1 and later:
// (1) an operand, starting with OperandToken0, defining which
//     s# register (D3D10_SB_OPERAND_TYPE_SAMPLER) is being declared.
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (s<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of samplers in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the s# is used in shader instructions, where the register
//     must be D3D10_SB_OPERAND_INDEX_DIMENSION_2D, and the meaning of the index 
//     dimensions are as follows: (s<id>[<idx>]):
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of sampler within space (may be dynamically indexed)
// (2) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------
typedef enum D3D10_SB_SAMPLER_MODE
{
    D3D10_SB_SAMPLER_MODE_DEFAULT      = 0,
    D3D10_SB_SAMPLER_MODE_COMPARISON   = 1,
    D3D10_SB_SAMPLER_MODE_MONO         = 2,
} D3D10_SB_SAMPLER_MODE;

#define D3D10_SB_SAMPLER_MODE_MASK  0x00007800
#define D3D10_SB_SAMPLER_MODE_SHIFT 11

// DECODER MACRO: Find out if a Constant Buffer is going to be indexed or not
#define DECODE_D3D10_SB_SAMPLER_MODE(OpcodeToken0) ((D3D10_SB_SAMPLER_MODE)(((OpcodeToken0)&D3D10_SB_SAMPLER_MODE_MASK)>>D3D10_SB_SAMPLER_MODE_SHIFT))

// ENCODER MACRO: Generate a resource return type for a component
#define ENCODE_D3D10_SB_SAMPLER_MODE(SamplerMode) (((SamplerMode)<<D3D10_SB_SAMPLER_MODE_SHIFT)&D3D10_SB_SAMPLER_MODE_MASK)

// ----------------------------------------------------------------------------
// Input Register Declaration (see separate declarations for Pixel Shaders)
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INPUT
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 1 operand:
// (1) Operand, starting with OperandToken0, defining which input
//     v# register (D3D10_SB_OPERAND_TYPE_INPUT) is being declared, 
//     including writemask.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Input Register Declaration w/System Interpreted Value
// (see separate declarations for Pixel Shaders)
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INPUT_SIV
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) Operand, starting with OperandToken0, defining which input
//     v# register (D3D10_SB_OPERAND_TYPE_INPUT) is being declared,
//     including writemask.  For Geometry Shaders, the input is 
//     v[vertex][attribute], and this declaration is only for which register 
//     on the attribute axis is being declared.  The vertex axis value must 
//     be equal to the # of vertices in the current input primitive for the GS
//     (i.e. 6 for triangle + adjacency).
// (2) a System Interpreted Value Name (NameToken)
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Input Register Declaration w/System Generated Value
// (available for all shaders incl. Pixel Shader, no interpolation mode needed)
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INPUT_SGV
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) Operand, starting with OperandToken0, defining which input
//     v# register (D3D10_SB_OPERAND_TYPE_INPUT) is being declared,
//     including writemask.
// (2) a System Generated Value Name (NameToken)
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Pixel Shader Input Register Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INPUT_PS
// [14:11] D3D10_SB_INTERPOLATION_MODE
// [23:15] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 1 operand:
// (1) Operand, starting with OperandToken0, defining which input
//     v# register (D3D10_SB_OPERAND_TYPE_INPUT) is being declared,
//     including writemask.
//
// ----------------------------------------------------------------------------
#define D3D10_SB_INPUT_INTERPOLATION_MODE_MASK  0x00007800
#define D3D10_SB_INPUT_INTERPOLATION_MODE_SHIFT 11

// DECODER MACRO: Find out interpolation mode for the input register
#define DECODE_D3D10_SB_INPUT_INTERPOLATION_MODE(OpcodeToken0) ((D3D10_SB_INTERPOLATION_MODE)(((OpcodeToken0)&D3D10_SB_INPUT_INTERPOLATION_MODE_MASK)>>D3D10_SB_INPUT_INTERPOLATION_MODE_SHIFT))

// ENCODER MACRO: Encode interpolation mode for a register.
#define ENCODE_D3D10_SB_INPUT_INTERPOLATION_MODE(InterpolationMode) (((InterpolationMode)<<D3D10_SB_INPUT_INTERPOLATION_MODE_SHIFT)&D3D10_SB_INPUT_INTERPOLATION_MODE_MASK)

// ----------------------------------------------------------------------------
// Pixel Shader Input Register Declaration w/System Interpreted Value
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INPUT_PS_SIV
// [14:11] D3D10_SB_INTERPOLATION_MODE
// [23:15] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) Operand, starting with OperandToken0, defining which input
//     v# register (D3D10_SB_OPERAND_TYPE_INPUT) is being declared.
// (2) a System Interpreted Value Name (NameToken)
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Pixel Shader Input Register Declaration w/System Generated Value
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INPUT_PS_SGV
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) Operand, starting with OperandToken0, defining which input
//     v# register (D3D10_SB_OPERAND_TYPE_INPUT) is being declared.
// (2) a System Generated Value Name (NameToken)
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Output Register Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_OUTPUT
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 1 operand:
// (1) Operand, starting with OperandToken0, defining which
//     o# register (D3D10_SB_OPERAND_TYPE_OUTPUT) is being declared,
//     including writemask.
//     (in Pixel Shader, output can also be one of 
//     D3D10_SB_OPERAND_TYPE_OUTPUT_DEPTH,
//     D3D11_SB_OPERAND_TYPE_OUTPUT_DEPTH_GREATER_EQUAL, or
//     D3D11_SB_OPERAND_TYPE_OUTPUT_DEPTH_LESS_EQUAL )
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Output Register Declaration w/System Interpreted Value
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_OUTPUT_SIV
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) an operand, starting with OperandToken0, defining which
//     o# register (D3D10_SB_OPERAND_TYPE_OUTPUT) is being declared,
//     including writemask.
// (2) a System Interpreted Name token (NameToken)
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Output Register Declaration w/System Generated Value
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_OUTPUT_SGV
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) an operand, starting with OperandToken0, defining which
//     o# register (D3D10_SB_OPERAND_TYPE_OUTPUT) is being declared,
//     including writemask.
// (2) a System Generated Name token (NameToken)
//
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// Input or Output Register Indexing Range Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INDEX_RANGE
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) an operand, starting with OperandToken0, defining which
//     input (v#) or output (o#) register is having its array indexing range
//     declared, including writemask.  For Geometry Shader inputs, 
//     it is assumed that the vertex axis is always fully indexable,
//     and 0 must be specified as the vertex# in this declaration, so that 
//     only the a range of attributes are having their index range defined.
//     
// (2) a DWORD representing the count of registers starting from the one
//     indicated in (1).
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Temp Register Declaration r0...r(n-1) 
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_TEMPS
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 1 operand:
// (1) DWORD (unsigned int) indicating how many temps are being declared.  
//     i.e. 5 means r0...r4 are declared.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Indexable Temp Register (x#[size]) Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 3 DWORDs:
// (1) Register index (defines which x# register is declared)
// (2) Number of registers in this register bank
// (3) Number of components in the array (1-4). 1 means .x, 2 means .xy etc.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Constant Buffer Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER
// [11]    D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN
// [23:12] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 1 operand on Shader Model 4.0 through 5.0:
// (1) Operand, starting with OperandToken0, defining which CB slot (cb#[size])
//     is being declared. (operand type: D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER)
//     The indexing dimension for the register must be 
//     D3D10_SB_OPERAND_INDEX_DIMENSION_2D, where the first index specifies
//     which cb#[] is being declared, and the second (array) index specifies the size 
//     of the buffer, as a count of 32-bit*4 elements.  (As opposed to when the 
//     cb#[] is used in shader instructions, and the array index represents which 
//     location in the constant buffer is being referenced.)
//     If the size is specified as 0, the CB size is not known (any size CB
//     can be bound to the slot).
//
// The order of constant buffer declarations in a shader indicates their
// relative priority from highest to lowest (hint to driver).
// 
// OpcodeToken0 is followed by 3 operands on Shader Model 5.1 and later:
// (1) Operand, starting with OperandToken0, defining which CB range (ID and bounds)
//     is being declared. (operand type: D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER)
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (cb<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of constant buffers in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the cb#[] is used in shader instructions: (cb<id>[<idx>][<loc>])
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of constant buffer within space (may be dynamically indexed)
//       3 <loc>: location of vector within constant buffer being referenced,
//          which may also be dynamically indexed, with no access pattern flag required.
// (2) a DWORD indicating the size of the constant buffer as a count of 16-byte vectors.
//     Each vector is 32-bit*4 elements == 128-bits == 16 bytes.
//     If the size is specified as 0, the CB size is not known (any size CB
//     can be bound to the slot).
// (3) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------

typedef enum D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN
{
    D3D10_SB_CONSTANT_BUFFER_IMMEDIATE_INDEXED  = 0,
    D3D10_SB_CONSTANT_BUFFER_DYNAMIC_INDEXED    = 1
} D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN;

#define D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN_MASK  0x00000800
#define D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN_SHIFT 11

// DECODER MACRO: Find out if a Constant Buffer is going to be indexed or not
#define DECODE_D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN(OpcodeToken0) ((D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN)(((OpcodeToken0)&D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN_MASK)>>D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN_SHIFT))

// ENCODER MACRO: Encode the access pattern for the Constant Buffer
#define ENCODE_D3D10_SB_D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN(AccessPattern) (((AccessPattern)<<D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN_SHIFT)&D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN_MASK)

// ----------------------------------------------------------------------------
// Immediate Constant Buffer Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_CUSTOMDATA
// [31:11] == D3D10_SB_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER
//
// OpcodeToken0 is followed by:
// (1) DWORD indicating length of declaration, including OpcodeToken0.
//     This length must = 2(for OpcodeToken0 and 1) + a multiple of 4 
//                                                    (# of immediate constants)
// (2) Sequence of 4-tuples of DWORDs defining the Immediate Constant Buffer.
//     The number of 4-tuples is (length above - 1) / 4
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Shader Message Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_CUSTOMDATA
// [31:11] == D3D11_SB_CUSTOMDATA_SHADER_MESSAGE
//
// OpcodeToken0 is followed by:
// (1) DWORD indicating length of declaration, including OpcodeToken0.
// (2) DWORD (D3D11_SB_SHADER_MESSAGE_ID) indicating shader message or error.
// (3) D3D11_SB_SHADER_MESSAGE_FORMAT indicating the convention for formatting the message.
// (4) DWORD indicating the number of characters in the string without the terminator.
// (5) DWORD indicating the number of operands.
// (6) DWORD indicating length of operands.
// (7) Encoded operands.
// (8) String with trailing zero, padded to a multiple of DWORDs.
//     The string is in the given format and the operands given should
//     be used for argument substitutions when formatting.
// ----------------------------------------------------------------------------

typedef enum D3D11_SB_SHADER_MESSAGE_ID
{
    D3D11_SB_SHADER_MESSAGE_ID_MESSAGE = 0x00200102,
    D3D11_SB_SHADER_MESSAGE_ID_ERROR = 0x00200103
} D3D11_SB_SHADER_MESSAGE_ID;

typedef enum D3D11_SB_SHADER_MESSAGE_FORMAT
{
    // No formatting, just a text string.  Operands are ignored.
    D3D11_SB_SHADER_MESSAGE_FORMAT_ANSI_TEXT,
    // Format string follows C/C++ printf conventions.
    D3D11_SB_SHADER_MESSAGE_FORMAT_ANSI_PRINTF,
} D3D11_SB_SHADER_MESSAGE_FORMAT;

// ----------------------------------------------------------------------------
// Shader Clip Plane Constant Mappings for DX9 hardware
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_CUSTOMDATA
// [31:11] == D3D11_SB_CUSTOMDATA_SHADER_CLIP_PLANE_CONSTANT_MAPPINGS_FOR_DX9
//
// OpcodeToken0 is followed by:
// (1) DWORD indicating length of declaration, including OpcodeToken0.
// (2) DWORD indicating number of constant mappings (up to 6 mappings).
// (3+) Constant mapping tables in following format.
//
// struct _Clip_Plane_Constant_Mapping
// {
//     WORD ConstantBufferIndex;  // cb[n]
//     WORD StartConstantElement; // starting index of cb[n][m]
//     WORD ConstantElemntCount;  // number of elements cb[n][m] ~ cb[n][m+l]
//     WORD Reserved;             //
// };
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Geometry Shader Input Primitive Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_GS_INPUT_PRIMITIVE
// [16:11] D3D10_SB_PRIMITIVE [not D3D10_SB_PRIMITIVE_TOPOLOGY]
// [23:17] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token. == 1
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// ----------------------------------------------------------------------------

#define D3D10_SB_GS_INPUT_PRIMITIVE_MASK  0x0001f800
#define D3D10_SB_GS_INPUT_PRIMITIVE_SHIFT 11

// DECODER MACRO: Given a primitive topology declaration,
// (OpcodeToken0), determine the primitive topology
// (D3D10_SB_PRIMITIVE enum)
#define DECODE_D3D10_SB_GS_INPUT_PRIMITIVE(OpcodeToken0) ((D3D10_SB_PRIMITIVE)(((OpcodeToken0)&D3D10_SB_GS_INPUT_PRIMITIVE_MASK)>>D3D10_SB_GS_INPUT_PRIMITIVE_SHIFT))

// ENCODER MACRO: Store primitive topology
// (D3D10_SB_PRIMITIVE enum) into a
// a primitive topology declaration token (OpcodeToken0)
#define ENCODE_D3D10_SB_GS_INPUT_PRIMITIVE(Prim) (((Prim)<<D3D10_SB_GS_INPUT_PRIMITIVE_SHIFT)&D3D10_SB_GS_INPUT_PRIMITIVE_MASK)

// ----------------------------------------------------------------------------
// Geometry Shader Output Topology Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY
// [17:11] D3D10_SB_PRIMITIVE_TOPOLOGY
// [23:18] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token. == 1
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// ----------------------------------------------------------------------------

#define D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY_MASK  0x0001f800
#define D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY_SHIFT 11

// DECODER MACRO: Given a primitive topology declaration,
// (OpcodeToken0), determine the primitive topology
// (D3D10_SB_PRIMITIVE_TOPOLOGY enum)
#define DECODE_D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY(OpcodeToken0) ((D3D10_SB_PRIMITIVE_TOPOLOGY)(((OpcodeToken0)&D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY_MASK)>>D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY_SHIFT))

// ENCODER MACRO: Store primitive topology
// (D3D10_SB_PRIMITIVE_TOPOLOGY enum) into a
// a primitive topology declaration token (OpcodeToken0)
#define ENCODE_D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY(PrimTopology) (((PrimTopology)<<D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY_SHIFT)&D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY_MASK)

// ----------------------------------------------------------------------------
// Geometry Shader Maximum Output Vertex Count Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by a DWORD representing the
// maximum number of primitives that could be output
// by the Geometry Shader.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Geometry Shader Instance Count Declaration
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by a UINT32 representing the
// number of instances of the geometry shader program to execute.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Hull Shader Declaration Phase: HS/DS Input Control Point Count
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT
// [16:11] Control point count 
// [23:17] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token. == 1
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// ----------------------------------------------------------------------------
#define D3D11_SB_INPUT_CONTROL_POINT_COUNT_MASK  0x0001f800
#define D3D11_SB_INPUT_CONTROL_POINT_COUNT_SHIFT 11

// DECODER MACRO: Given an input control point count declaration token,
// (OpcodeToken0), determine the control point count
#define DECODE_D3D11_SB_INPUT_CONTROL_POINT_COUNT(OpcodeToken0) ((UINT)(((OpcodeToken0)&D3D11_SB_INPUT_CONTROL_POINT_COUNT_MASK)>>D3D11_SB_INPUT_CONTROL_POINT_COUNT_SHIFT))

// ENCODER MACRO: Store input control point count into a declaration token
#define ENCODE_D3D11_SB_INPUT_CONTROL_POINT_COUNT(Count) (((Count)<<D3D11_SB_INPUT_CONTROL_POINT_COUNT_SHIFT)&D3D11_SB_INPUT_CONTROL_POINT_COUNT_MASK)

// ----------------------------------------------------------------------------
// Hull Shader Declaration Phase: HS Output Control Point Count
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT
// [16:11] Control point count 
// [23:17] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token. == 1
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// ----------------------------------------------------------------------------
#define D3D11_SB_OUTPUT_CONTROL_POINT_COUNT_MASK  0x0001f800
#define D3D11_SB_OUTPUT_CONTROL_POINT_COUNT_SHIFT 11

// DECODER MACRO: Given an output control point count declaration token,
// (OpcodeToken0), determine the control point count
#define DECODE_D3D11_SB_OUTPUT_CONTROL_POINT_COUNT(OpcodeToken0) ((UINT)(((OpcodeToken0)&D3D11_SB_OUTPUT_CONTROL_POINT_COUNT_MASK)>>D3D11_SB_OUTPUT_CONTROL_POINT_COUNT_SHIFT))

// ENCODER MACRO: Store output control point count into a declaration token
#define ENCODE_D3D11_SB_OUTPUT_CONTROL_POINT_COUNT(Count) (((Count)<<D3D11_SB_OUTPUT_CONTROL_POINT_COUNT_SHIFT)&D3D11_SB_OUTPUT_CONTROL_POINT_COUNT_MASK)

// ----------------------------------------------------------------------------
// Hull Shader Declaration Phase: Tessellator Domain
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_TESS_DOMAIN
// [12:11] Domain
// [23:13] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token. == 1
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// ----------------------------------------------------------------------------
typedef enum D3D11_SB_TESSELLATOR_DOMAIN
{
    D3D11_SB_TESSELLATOR_DOMAIN_UNDEFINED = 0,
    D3D11_SB_TESSELLATOR_DOMAIN_ISOLINE   = 1,
    D3D11_SB_TESSELLATOR_DOMAIN_TRI       = 2,
    D3D11_SB_TESSELLATOR_DOMAIN_QUAD      = 3
} D3D11_SB_TESSELLATOR_DOMAIN;

#define D3D11_SB_TESS_DOMAIN_MASK  0x00001800
#define D3D11_SB_TESS_DOMAIN_SHIFT 11

// DECODER MACRO: Given a tessellator domain declaration,
// (OpcodeToken0), determine the domain
// (D3D11_SB_TESSELLATOR_DOMAIN enum)
#define DECODE_D3D11_SB_TESS_DOMAIN(OpcodeToken0) ((D3D11_SB_TESSELLATOR_DOMAIN)(((OpcodeToken0)&D3D11_SB_TESS_DOMAIN_MASK)>>D3D11_SB_TESS_DOMAIN_SHIFT))

// ENCODER MACRO: Store tessellator domain
// (D3D11_SB_TESSELLATOR_DOMAIN enum) into a
// a tessellator domain declaration token (OpcodeToken0)
#define ENCODE_D3D11_SB_TESS_DOMAIN(Domain) (((Domain)<<D3D11_SB_TESS_DOMAIN_SHIFT)&D3D11_SB_TESS_DOMAIN_MASK)

// ----------------------------------------------------------------------------
// Hull Shader Declaration Phase: Tessellator Partitioning
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_TESS_PARTITIONING
// [13:11] Partitioning
// [23:14] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token. == 1
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// ----------------------------------------------------------------------------
typedef enum D3D11_SB_TESSELLATOR_PARTITIONING
{
    D3D11_SB_TESSELLATOR_PARTITIONING_UNDEFINED       = 0,
    D3D11_SB_TESSELLATOR_PARTITIONING_INTEGER         = 1,
    D3D11_SB_TESSELLATOR_PARTITIONING_POW2            = 2,
    D3D11_SB_TESSELLATOR_PARTITIONING_FRACTIONAL_ODD  = 3,
    D3D11_SB_TESSELLATOR_PARTITIONING_FRACTIONAL_EVEN = 4
} D3D11_SB_TESSELLATOR_PARTITIONING;

#define D3D11_SB_TESS_PARTITIONING_MASK  0x00003800
#define D3D11_SB_TESS_PARTITIONING_SHIFT 11

// DECODER MACRO: Given a tessellator partitioning declaration,
// (OpcodeToken0), determine the domain
// (D3D11_SB_TESSELLATOR_PARTITIONING enum)
#define DECODE_D3D11_SB_TESS_PARTITIONING(OpcodeToken0) ((D3D11_SB_TESSELLATOR_PARTITIONING)(((OpcodeToken0)&D3D11_SB_TESS_PARTITIONING_MASK)>>D3D11_SB_TESS_PARTITIONING_SHIFT))

// ENCODER MACRO: Store tessellator partitioning
// (D3D11_SB_TESSELLATOR_PARTITIONING enum) into a
// a tessellator partitioning declaration token (OpcodeToken0)
#define ENCODE_D3D11_SB_TESS_PARTITIONING(Partitioning) (((Partitioning)<<D3D11_SB_TESS_PARTITIONING_SHIFT)&D3D11_SB_TESS_PARTITIONING_MASK)

// ----------------------------------------------------------------------------
// Hull Shader Declaration Phase: Tessellator Output Primitive
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE
// [13:11] Output Primitive
// [23:14] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token. == 1
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// ----------------------------------------------------------------------------
typedef enum D3D11_SB_TESSELLATOR_OUTPUT_PRIMITIVE
{
    D3D11_SB_TESSELLATOR_OUTPUT_UNDEFINED     = 0,
    D3D11_SB_TESSELLATOR_OUTPUT_POINT         = 1,
    D3D11_SB_TESSELLATOR_OUTPUT_LINE          = 2,
    D3D11_SB_TESSELLATOR_OUTPUT_TRIANGLE_CW   = 3,
    D3D11_SB_TESSELLATOR_OUTPUT_TRIANGLE_CCW  = 4
} D3D11_SB_TESSELLATOR_OUTPUT_PRIMITIVE;

#define D3D11_SB_TESS_OUTPUT_PRIMITIVE_MASK  0x00003800
#define D3D11_SB_TESS_OUTPUT_PRIMITIVE_SHIFT 11

// DECODER MACRO: Given a tessellator output primitive declaration,
// (OpcodeToken0), determine the domain
// (D3D11_SB_TESSELLATOR_OUTPUT_PRIMITIVE enum)
#define DECODE_D3D11_SB_TESS_OUTPUT_PRIMITIVE(OpcodeToken0) ((D3D11_SB_TESSELLATOR_OUTPUT_PRIMITIVE)(((OpcodeToken0)&D3D11_SB_TESS_OUTPUT_PRIMITIVE_MASK)>>D3D11_SB_TESS_OUTPUT_PRIMITIVE_SHIFT))

// ENCODER MACRO: Store tessellator output primitive
// (D3D11_SB_TESSELLATOR_OUTPUT_PRIMITIVE enum) into a
// a tessellator output primitive declaration token (OpcodeToken0)
#define ENCODE_D3D11_SB_TESS_OUTPUT_PRIMITIVE(OutputPrimitive) (((OutputPrimitive)<<D3D11_SB_TESS_OUTPUT_PRIMITIVE_SHIFT)&D3D11_SB_TESS_OUTPUT_PRIMITIVE_MASK)


// ----------------------------------------------------------------------------
// Hull Shader Declaration Phase: Hull Shader Max Tessfactor
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_HS_MAX_TESSFACTOR
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by a float32 representing the
// maximum TessFactor.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Hull Shader Declaration Phase: Hull Shader Fork Phase Instance Count
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by a UINT32 representing the
// number of instances of the current fork phase program to execute.
//
// ----------------------------------------------------------------------------

typedef enum D3D10_SB_INTERPOLATION_MODE
{
    D3D10_SB_INTERPOLATION_UNDEFINED = 0,
    D3D10_SB_INTERPOLATION_CONSTANT = 1,
    D3D10_SB_INTERPOLATION_LINEAR = 2,
    D3D10_SB_INTERPOLATION_LINEAR_CENTROID = 3,
    D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE = 4,
    D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE_CENTROID = 5,
    D3D10_SB_INTERPOLATION_LINEAR_SAMPLE = 6, // DX10.1
    D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE_SAMPLE = 7, // DX10.1
} D3D10_SB_INTERPOLATION_MODE;

// Keep PRIMITIVE_TOPOLOGY values in sync with earlier DX versions (HW consumes values directly).
typedef enum D3D10_SB_PRIMITIVE_TOPOLOGY
{
    D3D10_SB_PRIMITIVE_TOPOLOGY_UNDEFINED = 0,
    D3D10_SB_PRIMITIVE_TOPOLOGY_POINTLIST = 1,
    D3D10_SB_PRIMITIVE_TOPOLOGY_LINELIST = 2,
    D3D10_SB_PRIMITIVE_TOPOLOGY_LINESTRIP = 3,
    D3D10_SB_PRIMITIVE_TOPOLOGY_TRIANGLELIST = 4,
    D3D10_SB_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP = 5,
    // 6 is reserved for legacy triangle fans
    // Adjacency values should be equal to (0x8 & non-adjacency):
    D3D10_SB_PRIMITIVE_TOPOLOGY_LINELIST_ADJ = 10,
    D3D10_SB_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ = 11,
    D3D10_SB_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ = 12,
    D3D10_SB_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ = 13,
} D3D10_SB_PRIMITIVE_TOPOLOGY;

typedef enum D3D10_SB_PRIMITIVE
{
    D3D10_SB_PRIMITIVE_UNDEFINED = 0,
    D3D10_SB_PRIMITIVE_POINT = 1,
    D3D10_SB_PRIMITIVE_LINE = 2,
    D3D10_SB_PRIMITIVE_TRIANGLE = 3,
    // Adjacency values should be equal to (0x4 & non-adjacency):
    D3D10_SB_PRIMITIVE_LINE_ADJ = 6,
    D3D10_SB_PRIMITIVE_TRIANGLE_ADJ = 7,
    D3D11_SB_PRIMITIVE_1_CONTROL_POINT_PATCH = 8,
    D3D11_SB_PRIMITIVE_2_CONTROL_POINT_PATCH = 9,
    D3D11_SB_PRIMITIVE_3_CONTROL_POINT_PATCH = 10,
    D3D11_SB_PRIMITIVE_4_CONTROL_POINT_PATCH = 11,
    D3D11_SB_PRIMITIVE_5_CONTROL_POINT_PATCH = 12,
    D3D11_SB_PRIMITIVE_6_CONTROL_POINT_PATCH = 13,
    D3D11_SB_PRIMITIVE_7_CONTROL_POINT_PATCH = 14,
    D3D11_SB_PRIMITIVE_8_CONTROL_POINT_PATCH = 15,
    D3D11_SB_PRIMITIVE_9_CONTROL_POINT_PATCH = 16,
    D3D11_SB_PRIMITIVE_10_CONTROL_POINT_PATCH = 17,
    D3D11_SB_PRIMITIVE_11_CONTROL_POINT_PATCH = 18,
    D3D11_SB_PRIMITIVE_12_CONTROL_POINT_PATCH = 19,
    D3D11_SB_PRIMITIVE_13_CONTROL_POINT_PATCH = 20,
    D3D11_SB_PRIMITIVE_14_CONTROL_POINT_PATCH = 21,
    D3D11_SB_PRIMITIVE_15_CONTROL_POINT_PATCH = 22,
    D3D11_SB_PRIMITIVE_16_CONTROL_POINT_PATCH = 23,
    D3D11_SB_PRIMITIVE_17_CONTROL_POINT_PATCH = 24,
    D3D11_SB_PRIMITIVE_18_CONTROL_POINT_PATCH = 25,
    D3D11_SB_PRIMITIVE_19_CONTROL_POINT_PATCH = 26,
    D3D11_SB_PRIMITIVE_20_CONTROL_POINT_PATCH = 27,
    D3D11_SB_PRIMITIVE_21_CONTROL_POINT_PATCH = 28,
    D3D11_SB_PRIMITIVE_22_CONTROL_POINT_PATCH = 29,
    D3D11_SB_PRIMITIVE_23_CONTROL_POINT_PATCH = 30,
    D3D11_SB_PRIMITIVE_24_CONTROL_POINT_PATCH = 31,
    D3D11_SB_PRIMITIVE_25_CONTROL_POINT_PATCH = 32,
    D3D11_SB_PRIMITIVE_26_CONTROL_POINT_PATCH = 33,
    D3D11_SB_PRIMITIVE_27_CONTROL_POINT_PATCH = 34,
    D3D11_SB_PRIMITIVE_28_CONTROL_POINT_PATCH = 35,
    D3D11_SB_PRIMITIVE_29_CONTROL_POINT_PATCH = 36,
    D3D11_SB_PRIMITIVE_30_CONTROL_POINT_PATCH = 37,
    D3D11_SB_PRIMITIVE_31_CONTROL_POINT_PATCH = 38,
    D3D11_SB_PRIMITIVE_32_CONTROL_POINT_PATCH = 39,
} D3D10_SB_PRIMITIVE;

typedef enum D3D10_SB_COMPONENT_MASK
{
    D3D10_SB_COMPONENT_MASK_X = 1,
    D3D10_SB_COMPONENT_MASK_Y = 2,
    D3D10_SB_COMPONENT_MASK_Z = 4,
    D3D10_SB_COMPONENT_MASK_W = 8,
    D3D10_SB_COMPONENT_MASK_R = 1,
    D3D10_SB_COMPONENT_MASK_G = 2,
    D3D10_SB_COMPONENT_MASK_B = 4,
    D3D10_SB_COMPONENT_MASK_A = 8,
    D3D10_SB_COMPONENT_MASK_ALL = 15,
} D3D10_SB_COMPONENT_MASK;

typedef enum D3D10_SB_NAME
{
    D3D10_SB_NAME_UNDEFINED = 0,
    D3D10_SB_NAME_POSITION = 1,
    D3D10_SB_NAME_CLIP_DISTANCE = 2,
    D3D10_SB_NAME_CULL_DISTANCE = 3,
    D3D10_SB_NAME_RENDER_TARGET_ARRAY_INDEX = 4,
    D3D10_SB_NAME_VIEWPORT_ARRAY_INDEX = 5,
    D3D10_SB_NAME_VERTEX_ID = 6,
    D3D10_SB_NAME_PRIMITIVE_ID = 7,
    D3D10_SB_NAME_INSTANCE_ID = 8,
    D3D10_SB_NAME_IS_FRONT_FACE = 9,
    D3D10_SB_NAME_SAMPLE_INDEX = 10,
    // The following are added for D3D11
    D3D11_SB_NAME_FINAL_QUAD_U_EQ_0_EDGE_TESSFACTOR = 11, 
    D3D11_SB_NAME_FINAL_QUAD_V_EQ_0_EDGE_TESSFACTOR = 12, 
    D3D11_SB_NAME_FINAL_QUAD_U_EQ_1_EDGE_TESSFACTOR = 13, 
    D3D11_SB_NAME_FINAL_QUAD_V_EQ_1_EDGE_TESSFACTOR = 14, 
    D3D11_SB_NAME_FINAL_QUAD_U_INSIDE_TESSFACTOR = 15, 
    D3D11_SB_NAME_FINAL_QUAD_V_INSIDE_TESSFACTOR = 16, 
    D3D11_SB_NAME_FINAL_TRI_U_EQ_0_EDGE_TESSFACTOR = 17, 
    D3D11_SB_NAME_FINAL_TRI_V_EQ_0_EDGE_TESSFACTOR = 18, 
    D3D11_SB_NAME_FINAL_TRI_W_EQ_0_EDGE_TESSFACTOR = 19, 
    D3D11_SB_NAME_FINAL_TRI_INSIDE_TESSFACTOR = 20, 
    D3D11_SB_NAME_FINAL_LINE_DETAIL_TESSFACTOR = 21,
    D3D11_SB_NAME_FINAL_LINE_DENSITY_TESSFACTOR = 22,
    // The following are added for D3D12
    D3D12_SB_NAME_BARYCENTRICS = 23,
    D3D12_SB_NAME_SHADINGRATE = 24,
    D3D12_SB_NAME_CULLPRIMITIVE = 25,
} D3D10_SB_NAME;

typedef enum D3D10_SB_RESOURCE_DIMENSION
{
    D3D10_SB_RESOURCE_DIMENSION_UNKNOWN = 0,
    D3D10_SB_RESOURCE_DIMENSION_BUFFER = 1,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURE1D = 2,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURE2D = 3,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURE2DMS = 4,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURE3D = 5,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURECUBE = 6,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURE1DARRAY = 7,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURE2DARRAY = 8,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURE2DMSARRAY = 9,
    D3D10_SB_RESOURCE_DIMENSION_TEXTURECUBEARRAY = 10,
    D3D11_SB_RESOURCE_DIMENSION_RAW_BUFFER = 11,
    D3D11_SB_RESOURCE_DIMENSION_STRUCTURED_BUFFER = 12,
} D3D10_SB_RESOURCE_DIMENSION;

typedef enum D3D10_SB_RESOURCE_RETURN_TYPE
{
    D3D10_SB_RETURN_TYPE_UNORM = 1,
    D3D10_SB_RETURN_TYPE_SNORM = 2,
    D3D10_SB_RETURN_TYPE_SINT = 3,
    D3D10_SB_RETURN_TYPE_UINT = 4,
    D3D10_SB_RETURN_TYPE_FLOAT = 5,
    D3D10_SB_RETURN_TYPE_MIXED = 6,
    D3D11_SB_RETURN_TYPE_DOUBLE = 7,
    D3D11_SB_RETURN_TYPE_CONTINUED = 8,
    D3D11_SB_RETURN_TYPE_UNUSED = 9,
} D3D10_SB_RESOURCE_RETURN_TYPE;

typedef enum D3D10_SB_REGISTER_COMPONENT_TYPE
{
    D3D10_SB_REGISTER_COMPONENT_UNKNOWN = 0,
    D3D10_SB_REGISTER_COMPONENT_UINT32 = 1,
    D3D10_SB_REGISTER_COMPONENT_SINT32 = 2,
    D3D10_SB_REGISTER_COMPONENT_FLOAT32 = 3
} D3D10_SB_REGISTER_COMPONENT_TYPE;

typedef enum D3D10_SB_INSTRUCTION_RETURN_TYPE
{
    D3D10_SB_INSTRUCTION_RETURN_FLOAT      = 0,
    D3D10_SB_INSTRUCTION_RETURN_UINT       = 1
} D3D10_SB_INSTRUCTION_RETURN_TYPE;

#define D3D10_SB_INSTRUCTION_RETURN_TYPE_MASK  0x00001800
#define D3D10_SB_INSTRUCTION_RETURN_TYPE_SHIFT 11

// DECODER MACRO: For an OpcodeToken0 with the return type 
// determine the return type.
#define DECODE_D3D10_SB_INSTRUCTION_RETURN_TYPE(OpcodeToken0) ((D3D10_SB_INSTRUCTION_RETURN_TYPE)(((OpcodeToken0)&D3D10_SB_INSTRUCTION_RETURN_TYPE_MASK)>>D3D10_SB_INSTRUCTION_RETURN_TYPE_SHIFT))
// ENCODER MACRO: Encode the return type for instructions
// in the opcode specific control range of OpcodeToken0
#define ENCODE_D3D10_SB_INSTRUCTION_RETURN_TYPE(ReturnType) (((ReturnType)<<D3D10_SB_INSTRUCTION_RETURN_TYPE_SHIFT)&D3D10_SB_INSTRUCTION_RETURN_TYPE_MASK)

// ----------------------------------------------------------------------------
// Interface function body Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_FUNCTION_BODY
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  If it is extended, then
//         it contains the actual instruction length in DWORDs, since
//         it may not fit into 7 bits if enough operands are defined.
//
// OpcodeToken0 is followed by a DWORD that represents the function body
// identifier.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Interface function table Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_FUNCTION_TABLE
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  If it is extended, then
//         it contains the actual instruction length in DWORDs, since
//         it may not fit into 7 bits if enough functions are defined.
//
// OpcodeToken0 is followed by a DWORD that represents the function table
// identifier and another DWORD (TableLength) that gives the number of
// functions in the table.
//
// This is followed by TableLength DWORDs which are function body indices.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Interface Declaration
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_DCL_INTERFACE
// [11]    1 if the interface is indexed dynamically, 0 otherwise.
// [23:12] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  If it is extended, then
//         it contains the actual instruction length in DWORDs, since
//         it may not fit into 7 bits if enough types are used.
//
// OpcodeToken0 is followed by a DWORD that represents the interface
// identifier. Next is a DWORD that gives the expected function table
// length. Then another DWORD (OpcodeToken3) with the following layout:
//
// [15:00] TableLength, the number of types that implement this interface
// [31:16] ArrayLength, the number of interfaces that are defined in this array.
//
// This is followed by TableLength DWORDs which are function table
// identifiers, representing possible tables for a given interface.
//
// ----------------------------------------------------------------------------

#define D3D11_SB_INTERFACE_INDEXED_BIT_MASK  0x00000800
#define D3D11_SB_INTERFACE_INDEXED_BIT_SHIFT 11

#define D3D11_SB_INTERFACE_TABLE_LENGTH_MASK  0x0000ffff
#define D3D11_SB_INTERFACE_TABLE_LENGTH_SHIFT 0

#define D3D11_SB_INTERFACE_ARRAY_LENGTH_MASK  0xffff0000
#define D3D11_SB_INTERFACE_ARRAY_LENGTH_SHIFT 16

// get/set the indexed bit for an interface definition
#define DECODE_D3D11_SB_INTERFACE_INDEXED_BIT(OpcodeToken0) ((((OpcodeToken0)&D3D11_SB_INTERFACE_INDEXED_BIT_MASK)>>D3D11_SB_INTERFACE_INDEXED_BIT_SHIFT) ? true : false)
#define ENCODE_D3D11_SB_INTERFACE_INDEXED_BIT(IndexedBit) (((IndexedBit)<<D3D11_SB_INTERFACE_INDEXED_BIT_SHIFT)&D3D11_SB_INTERFACE_INDEXED_BIT_MASK)

// get/set the table length for an interface definition
#define DECODE_D3D11_SB_INTERFACE_TABLE_LENGTH(OpcodeToken0) ((UINT)(((OpcodeToken0)&D3D11_SB_INTERFACE_TABLE_LENGTH_MASK)>>D3D11_SB_INTERFACE_TABLE_LENGTH_SHIFT))
#define ENCODE_D3D11_SB_INTERFACE_TABLE_LENGTH(TableLength) (((TableLength)<<D3D11_SB_INTERFACE_TABLE_LENGTH_SHIFT)&D3D11_SB_INTERFACE_TABLE_LENGTH_MASK)

// get/set the array length for an interface definition
#define DECODE_D3D11_SB_INTERFACE_ARRAY_LENGTH(OpcodeToken0) ((UINT)(((OpcodeToken0)&D3D11_SB_INTERFACE_ARRAY_LENGTH_MASK)>>D3D11_SB_INTERFACE_ARRAY_LENGTH_SHIFT))
#define ENCODE_D3D11_SB_INTERFACE_ARRAY_LENGTH(ArrayLength) (((ArrayLength)<<D3D11_SB_INTERFACE_ARRAY_LENGTH_SHIFT)&D3D11_SB_INTERFACE_ARRAY_LENGTH_MASK)

// ----------------------------------------------------------------------------
// Interface call
//
// OpcodeToken0:
//
// [10:00] D3D10_SB_OPCODE_INTERFACE_CALL
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  If it is extended, then
//         it contains the actual instruction length in DWORDs, since
//         it may not fit into 7 bits if enough types are used.
//
// OpcodeToken0 is followed by a DWORD that gives the function index to
// call in the function table specified for the given interface. 
// Next is the interface operand.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Thread Group Declaration (Compute Shader)
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_THREAD_GROUP
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  If it is extended, then
//         it contains the actual instruction length in DWORDs, since
//         it may not fit into 7 bits if enough types are used.
//
// OpcodeToken0 is followed by 3 DWORDs, the Thread Group dimensions as UINT32:
// x, y, z
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Typed Unordered Access View Declaration
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED
// [15:11] D3D10_SB_RESOURCE_DIMENSION
// [16:16] D3D11_SB_GLOBALLY_COHERENT_ACCESS or 0 (LOCALLY_COHERENT)
// [17:17] D3D11_SB_RASTERIZER_ORDERED_ACCESS or 0
// [23:18] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands on Shader Models 4.0 through 5.0:
// (1) an operand, starting with OperandToken0, defining which
//     u# register (D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) is being declared.
// (2) a Resource Return Type token (ResourceReturnTypeToken)
//
// OpcodeToken0 is followed by 3 operands on Shader Model 5.1 and later:
// (1) an operand, starting with OperandToken0, defining which
//     u# register (D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) is being declared.
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (u<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of UAV's in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the u# is used in shader instructions, where the register
//     must be D3D10_SB_OPERAND_INDEX_DIMENSION_2D, and the meaning of the index 
//     dimensions are as follows: (u<id>[<idx>]):
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of uav within space (may be dynamically indexed)
// (2) a Resource Return Type token (ResourceReturnTypeToken)
// (3) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------
// UAV access scope flags
#define D3D11_SB_GLOBALLY_COHERENT_ACCESS 0x00010000
#define D3D11_SB_ACCESS_COHERENCY_MASK    0x00010000

// DECODER MACRO: Retrieve flags for sync instruction from OpcodeToken0.
#define DECODE_D3D11_SB_ACCESS_COHERENCY_FLAGS(OperandToken0) ((OperandToken0)&D3D11_SB_ACCESS_COHERENCY_MASK)

// ENCODER MACRO: Given a set of sync instruciton flags, encode them in OpcodeToken0.
#define ENCODE_D3D11_SB_ACCESS_COHERENCY_FLAGS(Flags) ((Flags)&D3D11_SB_ACCESS_COHERENCY_MASK)

// Additional UAV access flags
#define D3D11_SB_RASTERIZER_ORDERED_ACCESS 0x00020000

// Resource flags mask.  Use to retrieve all resource flags, including the order preserving counter.
#define D3D11_SB_RESOURCE_FLAGS_MASK    (D3D11_SB_GLOBALLY_COHERENT_ACCESS|D3D11_SB_RASTERIZER_ORDERED_ACCESS|D3D11_SB_UAV_HAS_ORDER_PRESERVING_COUNTER)

// DECODER MACRO: Retrieve UAV access flags for from OpcodeToken0.
#define DECODE_D3D11_SB_RESOURCE_FLAGS(OperandToken0) ((OperandToken0)&D3D11_SB_RESOURCE_FLAGS_MASK)

// ENCODER MACRO: Given UAV access flags, encode them in OpcodeToken0.
#define ENCODE_D3D11_SB_RESOURCE_FLAGS(Flags) ((Flags)&D3D11_SB_RESOURCE_FLAGS_MASK)

// ----------------------------------------------------------------------------
// Raw Unordered Access View Declaration
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_RAW
// [15:11] Ignored, 0
// [16:16] D3D11_SB_GLOBALLY_COHERENT_ACCESS or 0 (LOCALLY_COHERENT)
// [17:17] D3D11_SB_RASTERIZER_ORDERED_ACCESS or 0
// [23:18] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 1 operand on Shader Models 4.0 through 5.0:
// (1) an operand, starting with OperandToken0, defining which
//     u# register (D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) is being declared.
//
// OpcodeToken0 is followed by 2 operands on Shader Model 5.1 and later:
// (1) an operand, starting with OperandToken0, defining which
//     u# register (D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) is being declared.
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (u<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of UAV's in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the u# is used in shader instructions, where the register
//     must be D3D10_SB_OPERAND_INDEX_DIMENSION_2D, and the meaning of the index 
//     dimensions are as follows: (u<id>[<idx>]):
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of uav within space (may be dynamically indexed)
// (2) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Structured Unordered Access View Declaration
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_STRUCTURED
// [15:11] Ignored, 0
// [16:16] D3D11_SB_GLOBALLY_COHERENT_ACCESS or 0 (LOCALLY_COHERENT)
// [17:17] D3D11_SB_RASTERIZER_ORDERED_ACCESS or 0
// [22:18] Ignored, 0
// [23:23] D3D11_SB_UAV_HAS_ORDER_PRESERVING_COUNTER or 0
//
//            The presence of this flag means that if a UAV is bound to the
//            corresponding slot, it must have been created with 
//            D3D11_BUFFER_UAV_FLAG_COUNTER at the API.  Also, the shader
//            can contain either imm_atomic_alloc or _consume instructions
//            operating on the given UAV.
// 
//            If this flag is not present, the shader can still contain
//            either imm_atomic_alloc or imm_atomic_consume instructions for
//            this UAV.  But if such instructions are present in this case,
//            and a UAV is bound corresponding slot, it must have been created 
//            with the D3D11_BUFFER_UAV_FLAG_APPEND flag at the API.
//            Append buffers have a counter as well, but values returned 
//            to the shader are only valid for the lifetime of the shader 
//            invocation.
//
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) an operand, starting with OperandToken0, defining which
//     u# register (D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) is 
//     being declared.
// (2) a DWORD indicating UINT32 byte stride
//
// OpcodeToken0 is followed by 3 operands on Shader Model 5.1 and later:
// (1) an operand, starting with OperandToken0, defining which
//     u# register (D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) is being declared.
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (u<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of UAV's in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the u# is used in shader instructions, where the register
//     must be D3D10_SB_OPERAND_INDEX_DIMENSION_2D, and the meaning of the index 
//     dimensions are as follows: (u<id>[<idx>]):
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of uav within space (may be dynamically indexed)
// (2) a DWORD indicating UINT32 byte stride
// (3) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------
// UAV flags
#define D3D11_SB_UAV_HAS_ORDER_PRESERVING_COUNTER 0x00800000
#define D3D11_SB_UAV_FLAGS_MASK                   0x00800000

// DECODER MACRO: Retrieve flags about UAV from OpcodeToken0.
#define DECODE_D3D11_SB_UAV_FLAGS(OperandToken0) ((OperandToken0)&D3D11_SB_UAV_FLAGS_MASK)

// ENCODER MACRO: Given a set of UAV flags, encode them in OpcodeToken0.
#define ENCODE_D3D11_SB_UAV_FLAGS(Flags) ((Flags)&D3D11_SB_UAV_FLAGS_MASK)

// ----------------------------------------------------------------------------
// Raw Thread Group Shared Memory Declaration
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_THREAD_GROUP_SHARED_MEMORY_RAW
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) an operand, starting with OperandToken0, defining which
//     g# register (D3D11_SB_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY) is being declared.
// (2) a DWORD indicating the byte count, which must be a multiple of 4.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Structured Thread Group Shared Memory Declaration
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_THREAD_GROUP_SHARED_MEMORY_STRUCTURED
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 3 operands:
// (1) an operand, starting with OperandToken0, defining which
//     g# register (D3D11_SB_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY) is 
//     being declared.
// (2) a DWORD indicating UINT32 struct byte stride
// (3) a DWORD indicating UINT32 struct count
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Raw Shader Resource View Declaration
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_RESOURCE_RAW
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 1 operand:
// (1) an operand, starting with OperandToken0, defining which
//     t# register (D3D10_SB_OPERAND_TYPE_RESOURCE) is being declared.
//
// OpcodeToken0 is followed by 2 operands on Shader Model 5.1 and later:
// (1) an operand, starting with OperandToken0, defining which
//     t# register (D3D10_SB_OPERAND_TYPE_RESOURCE) is being declared.
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (t<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of resources in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the t# is used in shader instructions, where the register
//     must be D3D10_SB_OPERAND_INDEX_DIMENSION_2D, and the meaning of the index 
//     dimensions are as follows: (t<id>[<idx>]):
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of resource within space (may be dynamically indexed)
// (2) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Structured Shader Resource View Declaration
//
// OpcodeToken0:
//
// [10:00] D3D11_SB_OPCODE_DCL_RESOURCE_STRUCTURED
// [23:11] Ignored, 0
// [30:24] Instruction length in DWORDs including the opcode token.
// [31]    0 normally. 1 if extended operand definition, meaning next DWORD
//         contains extended operand description.  This dcl is currently not
//         extended.
//
// OpcodeToken0 is followed by 2 operands:
// (1) an operand, starting with OperandToken0, defining which
//     g# register (D3D10_SB_OPERAND_TYPE_RESOURCE) is 
//     being declared.
// (2) a DWORD indicating UINT32 struct byte stride
//
// OpcodeToken0 is followed by 3 operands on Shader Model 5.1 and later:
// (1) an operand, starting with OperandToken0, defining which
//     t# register (D3D10_SB_OPERAND_TYPE_RESOURCE) is being declared.
//     The indexing dimension for the register must be D3D10_SB_OPERAND_INDEX_DIMENSION_3D, 
//     and the meaning of the index dimensions are as follows: (t<id>[<lbound>:<ubound>])
//       1 <id>:     variable ID being declared
//       2 <lbound>: the lower bound of the range of resources in the space
//       3 <ubound>: the upper bound (inclusive) of this range
//     As opposed to when the t# is used in shader instructions, where the register
//     must be D3D10_SB_OPERAND_INDEX_DIMENSION_2D, and the meaning of the index 
//     dimensions are as follows: (t<id>[<idx>]):
//       1 <id>:  variable ID being used (matches dcl)
//       2 <idx>: absolute index of resource within space (may be dynamically indexed)
// (2) a DWORD indicating UINT32 struct byte stride
// (3) a DWORD indicating the space index.
//
// ----------------------------------------------------------------------------

#endif /* WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES) */
#pragma endregion
