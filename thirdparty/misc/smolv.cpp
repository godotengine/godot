// smol-v - public domain - https://github.com/aras-p/smol-v
// authored 2016-2020 by Aras Pranckevicius
// no warranty implied; use at your own risk
// See end of file for license information.

#include "smolv.h"
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>

#if !defined(_MSC_VER) && __cplusplus < 201103L
#define static_assert(x,y)
#endif

#define _SMOLV_ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))

// --------------------------------------------------------------------------------------------
// Metadata about known SPIR-V operations

enum SpvOp
{
	SpvOpNop = 0,
	SpvOpUndef = 1,
	SpvOpSourceContinued = 2,
	SpvOpSource = 3,
	SpvOpSourceExtension = 4,
	SpvOpName = 5,
	SpvOpMemberName = 6,
	SpvOpString = 7,
	SpvOpLine = 8,
	SpvOpExtension = 10,
	SpvOpExtInstImport = 11,
	SpvOpExtInst = 12,
	SpvOpVectorShuffleCompact = 13, // not in SPIR-V, added for SMOL-V!
	SpvOpMemoryModel = 14,
	SpvOpEntryPoint = 15,
	SpvOpExecutionMode = 16,
	SpvOpCapability = 17,
	SpvOpTypeVoid = 19,
	SpvOpTypeBool = 20,
	SpvOpTypeInt = 21,
	SpvOpTypeFloat = 22,
	SpvOpTypeVector = 23,
	SpvOpTypeMatrix = 24,
	SpvOpTypeImage = 25,
	SpvOpTypeSampler = 26,
	SpvOpTypeSampledImage = 27,
	SpvOpTypeArray = 28,
	SpvOpTypeRuntimeArray = 29,
	SpvOpTypeStruct = 30,
	SpvOpTypeOpaque = 31,
	SpvOpTypePointer = 32,
	SpvOpTypeFunction = 33,
	SpvOpTypeEvent = 34,
	SpvOpTypeDeviceEvent = 35,
	SpvOpTypeReserveId = 36,
	SpvOpTypeQueue = 37,
	SpvOpTypePipe = 38,
	SpvOpTypeForwardPointer = 39,
	SpvOpConstantTrue = 41,
	SpvOpConstantFalse = 42,
	SpvOpConstant = 43,
	SpvOpConstantComposite = 44,
	SpvOpConstantSampler = 45,
	SpvOpConstantNull = 46,
	SpvOpSpecConstantTrue = 48,
	SpvOpSpecConstantFalse = 49,
	SpvOpSpecConstant = 50,
	SpvOpSpecConstantComposite = 51,
	SpvOpSpecConstantOp = 52,
	SpvOpFunction = 54,
	SpvOpFunctionParameter = 55,
	SpvOpFunctionEnd = 56,
	SpvOpFunctionCall = 57,
	SpvOpVariable = 59,
	SpvOpImageTexelPointer = 60,
	SpvOpLoad = 61,
	SpvOpStore = 62,
	SpvOpCopyMemory = 63,
	SpvOpCopyMemorySized = 64,
	SpvOpAccessChain = 65,
	SpvOpInBoundsAccessChain = 66,
	SpvOpPtrAccessChain = 67,
	SpvOpArrayLength = 68,
	SpvOpGenericPtrMemSemantics = 69,
	SpvOpInBoundsPtrAccessChain = 70,
	SpvOpDecorate = 71,
	SpvOpMemberDecorate = 72,
	SpvOpDecorationGroup = 73,
	SpvOpGroupDecorate = 74,
	SpvOpGroupMemberDecorate = 75,
	SpvOpVectorExtractDynamic = 77,
	SpvOpVectorInsertDynamic = 78,
	SpvOpVectorShuffle = 79,
	SpvOpCompositeConstruct = 80,
	SpvOpCompositeExtract = 81,
	SpvOpCompositeInsert = 82,
	SpvOpCopyObject = 83,
	SpvOpTranspose = 84,
	SpvOpSampledImage = 86,
	SpvOpImageSampleImplicitLod = 87,
	SpvOpImageSampleExplicitLod = 88,
	SpvOpImageSampleDrefImplicitLod = 89,
	SpvOpImageSampleDrefExplicitLod = 90,
	SpvOpImageSampleProjImplicitLod = 91,
	SpvOpImageSampleProjExplicitLod = 92,
	SpvOpImageSampleProjDrefImplicitLod = 93,
	SpvOpImageSampleProjDrefExplicitLod = 94,
	SpvOpImageFetch = 95,
	SpvOpImageGather = 96,
	SpvOpImageDrefGather = 97,
	SpvOpImageRead = 98,
	SpvOpImageWrite = 99,
	SpvOpImage = 100,
	SpvOpImageQueryFormat = 101,
	SpvOpImageQueryOrder = 102,
	SpvOpImageQuerySizeLod = 103,
	SpvOpImageQuerySize = 104,
	SpvOpImageQueryLod = 105,
	SpvOpImageQueryLevels = 106,
	SpvOpImageQuerySamples = 107,
	SpvOpConvertFToU = 109,
	SpvOpConvertFToS = 110,
	SpvOpConvertSToF = 111,
	SpvOpConvertUToF = 112,
	SpvOpUConvert = 113,
	SpvOpSConvert = 114,
	SpvOpFConvert = 115,
	SpvOpQuantizeToF16 = 116,
	SpvOpConvertPtrToU = 117,
	SpvOpSatConvertSToU = 118,
	SpvOpSatConvertUToS = 119,
	SpvOpConvertUToPtr = 120,
	SpvOpPtrCastToGeneric = 121,
	SpvOpGenericCastToPtr = 122,
	SpvOpGenericCastToPtrExplicit = 123,
	SpvOpBitcast = 124,
	SpvOpSNegate = 126,
	SpvOpFNegate = 127,
	SpvOpIAdd = 128,
	SpvOpFAdd = 129,
	SpvOpISub = 130,
	SpvOpFSub = 131,
	SpvOpIMul = 132,
	SpvOpFMul = 133,
	SpvOpUDiv = 134,
	SpvOpSDiv = 135,
	SpvOpFDiv = 136,
	SpvOpUMod = 137,
	SpvOpSRem = 138,
	SpvOpSMod = 139,
	SpvOpFRem = 140,
	SpvOpFMod = 141,
	SpvOpVectorTimesScalar = 142,
	SpvOpMatrixTimesScalar = 143,
	SpvOpVectorTimesMatrix = 144,
	SpvOpMatrixTimesVector = 145,
	SpvOpMatrixTimesMatrix = 146,
	SpvOpOuterProduct = 147,
	SpvOpDot = 148,
	SpvOpIAddCarry = 149,
	SpvOpISubBorrow = 150,
	SpvOpUMulExtended = 151,
	SpvOpSMulExtended = 152,
	SpvOpAny = 154,
	SpvOpAll = 155,
	SpvOpIsNan = 156,
	SpvOpIsInf = 157,
	SpvOpIsFinite = 158,
	SpvOpIsNormal = 159,
	SpvOpSignBitSet = 160,
	SpvOpLessOrGreater = 161,
	SpvOpOrdered = 162,
	SpvOpUnordered = 163,
	SpvOpLogicalEqual = 164,
	SpvOpLogicalNotEqual = 165,
	SpvOpLogicalOr = 166,
	SpvOpLogicalAnd = 167,
	SpvOpLogicalNot = 168,
	SpvOpSelect = 169,
	SpvOpIEqual = 170,
	SpvOpINotEqual = 171,
	SpvOpUGreaterThan = 172,
	SpvOpSGreaterThan = 173,
	SpvOpUGreaterThanEqual = 174,
	SpvOpSGreaterThanEqual = 175,
	SpvOpULessThan = 176,
	SpvOpSLessThan = 177,
	SpvOpULessThanEqual = 178,
	SpvOpSLessThanEqual = 179,
	SpvOpFOrdEqual = 180,
	SpvOpFUnordEqual = 181,
	SpvOpFOrdNotEqual = 182,
	SpvOpFUnordNotEqual = 183,
	SpvOpFOrdLessThan = 184,
	SpvOpFUnordLessThan = 185,
	SpvOpFOrdGreaterThan = 186,
	SpvOpFUnordGreaterThan = 187,
	SpvOpFOrdLessThanEqual = 188,
	SpvOpFUnordLessThanEqual = 189,
	SpvOpFOrdGreaterThanEqual = 190,
	SpvOpFUnordGreaterThanEqual = 191,
	SpvOpShiftRightLogical = 194,
	SpvOpShiftRightArithmetic = 195,
	SpvOpShiftLeftLogical = 196,
	SpvOpBitwiseOr = 197,
	SpvOpBitwiseXor = 198,
	SpvOpBitwiseAnd = 199,
	SpvOpNot = 200,
	SpvOpBitFieldInsert = 201,
	SpvOpBitFieldSExtract = 202,
	SpvOpBitFieldUExtract = 203,
	SpvOpBitReverse = 204,
	SpvOpBitCount = 205,
	SpvOpDPdx = 207,
	SpvOpDPdy = 208,
	SpvOpFwidth = 209,
	SpvOpDPdxFine = 210,
	SpvOpDPdyFine = 211,
	SpvOpFwidthFine = 212,
	SpvOpDPdxCoarse = 213,
	SpvOpDPdyCoarse = 214,
	SpvOpFwidthCoarse = 215,
	SpvOpEmitVertex = 218,
	SpvOpEndPrimitive = 219,
	SpvOpEmitStreamVertex = 220,
	SpvOpEndStreamPrimitive = 221,
	SpvOpControlBarrier = 224,
	SpvOpMemoryBarrier = 225,
	SpvOpAtomicLoad = 227,
	SpvOpAtomicStore = 228,
	SpvOpAtomicExchange = 229,
	SpvOpAtomicCompareExchange = 230,
	SpvOpAtomicCompareExchangeWeak = 231,
	SpvOpAtomicIIncrement = 232,
	SpvOpAtomicIDecrement = 233,
	SpvOpAtomicIAdd = 234,
	SpvOpAtomicISub = 235,
	SpvOpAtomicSMin = 236,
	SpvOpAtomicUMin = 237,
	SpvOpAtomicSMax = 238,
	SpvOpAtomicUMax = 239,
	SpvOpAtomicAnd = 240,
	SpvOpAtomicOr = 241,
	SpvOpAtomicXor = 242,
	SpvOpPhi = 245,
	SpvOpLoopMerge = 246,
	SpvOpSelectionMerge = 247,
	SpvOpLabel = 248,
	SpvOpBranch = 249,
	SpvOpBranchConditional = 250,
	SpvOpSwitch = 251,
	SpvOpKill = 252,
	SpvOpReturn = 253,
	SpvOpReturnValue = 254,
	SpvOpUnreachable = 255,
	SpvOpLifetimeStart = 256,
	SpvOpLifetimeStop = 257,
	SpvOpGroupAsyncCopy = 259,
	SpvOpGroupWaitEvents = 260,
	SpvOpGroupAll = 261,
	SpvOpGroupAny = 262,
	SpvOpGroupBroadcast = 263,
	SpvOpGroupIAdd = 264,
	SpvOpGroupFAdd = 265,
	SpvOpGroupFMin = 266,
	SpvOpGroupUMin = 267,
	SpvOpGroupSMin = 268,
	SpvOpGroupFMax = 269,
	SpvOpGroupUMax = 270,
	SpvOpGroupSMax = 271,
	SpvOpReadPipe = 274,
	SpvOpWritePipe = 275,
	SpvOpReservedReadPipe = 276,
	SpvOpReservedWritePipe = 277,
	SpvOpReserveReadPipePackets = 278,
	SpvOpReserveWritePipePackets = 279,
	SpvOpCommitReadPipe = 280,
	SpvOpCommitWritePipe = 281,
	SpvOpIsValidReserveId = 282,
	SpvOpGetNumPipePackets = 283,
	SpvOpGetMaxPipePackets = 284,
	SpvOpGroupReserveReadPipePackets = 285,
	SpvOpGroupReserveWritePipePackets = 286,
	SpvOpGroupCommitReadPipe = 287,
	SpvOpGroupCommitWritePipe = 288,
	SpvOpEnqueueMarker = 291,
	SpvOpEnqueueKernel = 292,
	SpvOpGetKernelNDrangeSubGroupCount = 293,
	SpvOpGetKernelNDrangeMaxSubGroupSize = 294,
	SpvOpGetKernelWorkGroupSize = 295,
	SpvOpGetKernelPreferredWorkGroupSizeMultiple = 296,
	SpvOpRetainEvent = 297,
	SpvOpReleaseEvent = 298,
	SpvOpCreateUserEvent = 299,
	SpvOpIsValidEvent = 300,
	SpvOpSetUserEventStatus = 301,
	SpvOpCaptureEventProfilingInfo = 302,
	SpvOpGetDefaultQueue = 303,
	SpvOpBuildNDRange = 304,
	SpvOpImageSparseSampleImplicitLod = 305,
	SpvOpImageSparseSampleExplicitLod = 306,
	SpvOpImageSparseSampleDrefImplicitLod = 307,
	SpvOpImageSparseSampleDrefExplicitLod = 308,
	SpvOpImageSparseSampleProjImplicitLod = 309,
	SpvOpImageSparseSampleProjExplicitLod = 310,
	SpvOpImageSparseSampleProjDrefImplicitLod = 311,
	SpvOpImageSparseSampleProjDrefExplicitLod = 312,
	SpvOpImageSparseFetch = 313,
	SpvOpImageSparseGather = 314,
	SpvOpImageSparseDrefGather = 315,
	SpvOpImageSparseTexelsResident = 316,
	SpvOpNoLine = 317,
	SpvOpAtomicFlagTestAndSet = 318,
	SpvOpAtomicFlagClear = 319,
	SpvOpImageSparseRead = 320,
	SpvOpSizeOf = 321,
	SpvOpTypePipeStorage = 322,
	SpvOpConstantPipeStorage = 323,
	SpvOpCreatePipeFromPipeStorage = 324,
	SpvOpGetKernelLocalSizeForSubgroupCount = 325,
	SpvOpGetKernelMaxNumSubgroups = 326,
	SpvOpTypeNamedBarrier = 327,
	SpvOpNamedBarrierInitialize = 328,
	SpvOpMemoryNamedBarrier = 329,
	SpvOpModuleProcessed = 330,
	SpvOpExecutionModeId = 331,
	SpvOpDecorateId = 332,
	SpvOpGroupNonUniformElect = 333,
	SpvOpGroupNonUniformAll = 334,
	SpvOpGroupNonUniformAny = 335,
	SpvOpGroupNonUniformAllEqual = 336,
	SpvOpGroupNonUniformBroadcast = 337,
	SpvOpGroupNonUniformBroadcastFirst = 338,
	SpvOpGroupNonUniformBallot = 339,
	SpvOpGroupNonUniformInverseBallot = 340,
	SpvOpGroupNonUniformBallotBitExtract = 341,
	SpvOpGroupNonUniformBallotBitCount = 342,
	SpvOpGroupNonUniformBallotFindLSB = 343,
	SpvOpGroupNonUniformBallotFindMSB = 344,
	SpvOpGroupNonUniformShuffle = 345,
	SpvOpGroupNonUniformShuffleXor = 346,
	SpvOpGroupNonUniformShuffleUp = 347,
	SpvOpGroupNonUniformShuffleDown = 348,
	SpvOpGroupNonUniformIAdd = 349,
	SpvOpGroupNonUniformFAdd = 350,
	SpvOpGroupNonUniformIMul = 351,
	SpvOpGroupNonUniformFMul = 352,
	SpvOpGroupNonUniformSMin = 353,
	SpvOpGroupNonUniformUMin = 354,
	SpvOpGroupNonUniformFMin = 355,
	SpvOpGroupNonUniformSMax = 356,
	SpvOpGroupNonUniformUMax = 357,
	SpvOpGroupNonUniformFMax = 358,
	SpvOpGroupNonUniformBitwiseAnd = 359,
	SpvOpGroupNonUniformBitwiseOr = 360,
	SpvOpGroupNonUniformBitwiseXor = 361,
	SpvOpGroupNonUniformLogicalAnd = 362,
	SpvOpGroupNonUniformLogicalOr = 363,
	SpvOpGroupNonUniformLogicalXor = 364,
	SpvOpGroupNonUniformQuadBroadcast = 365,
	SpvOpGroupNonUniformQuadSwap = 366,
};
static const int kKnownOpsCount = SpvOpGroupNonUniformQuadSwap+1;


static const char* kSpirvOpNames[] =
{
	"Nop",
	"Undef",
	"SourceContinued",
	"Source",
	"SourceExtension",
	"Name",
	"MemberName",
	"String",
	"Line",
	"#9",
	"Extension",
	"ExtInstImport",
	"ExtInst",
	"VectorShuffleCompact",
	"MemoryModel",
	"EntryPoint",
	"ExecutionMode",
	"Capability",
	"#18",
	"TypeVoid",
	"TypeBool",
	"TypeInt",
	"TypeFloat",
	"TypeVector",
	"TypeMatrix",
	"TypeImage",
	"TypeSampler",
	"TypeSampledImage",
	"TypeArray",
	"TypeRuntimeArray",
	"TypeStruct",
	"TypeOpaque",
	"TypePointer",
	"TypeFunction",
	"TypeEvent",
	"TypeDeviceEvent",
	"TypeReserveId",
	"TypeQueue",
	"TypePipe",
	"TypeForwardPointer",
	"#40",
	"ConstantTrue",
	"ConstantFalse",
	"Constant",
	"ConstantComposite",
	"ConstantSampler",
	"ConstantNull",
	"#47",
	"SpecConstantTrue",
	"SpecConstantFalse",
	"SpecConstant",
	"SpecConstantComposite",
	"SpecConstantOp",
	"#53",
	"Function",
	"FunctionParameter",
	"FunctionEnd",
	"FunctionCall",
	"#58",
	"Variable",
	"ImageTexelPointer",
	"Load",
	"Store",
	"CopyMemory",
	"CopyMemorySized",
	"AccessChain",
	"InBoundsAccessChain",
	"PtrAccessChain",
	"ArrayLength",
	"GenericPtrMemSemantics",
	"InBoundsPtrAccessChain",
	"Decorate",
	"MemberDecorate",
	"DecorationGroup",
	"GroupDecorate",
	"GroupMemberDecorate",
	"#76",
	"VectorExtractDynamic",
	"VectorInsertDynamic",
	"VectorShuffle",
	"CompositeConstruct",
	"CompositeExtract",
	"CompositeInsert",
	"CopyObject",
	"Transpose",
	"#85",
	"SampledImage",
	"ImageSampleImplicitLod",
	"ImageSampleExplicitLod",
	"ImageSampleDrefImplicitLod",
	"ImageSampleDrefExplicitLod",
	"ImageSampleProjImplicitLod",
	"ImageSampleProjExplicitLod",
	"ImageSampleProjDrefImplicitLod",
	"ImageSampleProjDrefExplicitLod",
	"ImageFetch",
	"ImageGather",
	"ImageDrefGather",
	"ImageRead",
	"ImageWrite",
	"Image",
	"ImageQueryFormat",
	"ImageQueryOrder",
	"ImageQuerySizeLod",
	"ImageQuerySize",
	"ImageQueryLod",
	"ImageQueryLevels",
	"ImageQuerySamples",
	"#108",
	"ConvertFToU",
	"ConvertFToS",
	"ConvertSToF",
	"ConvertUToF",
	"UConvert",
	"SConvert",
	"FConvert",
	"QuantizeToF16",
	"ConvertPtrToU",
	"SatConvertSToU",
	"SatConvertUToS",
	"ConvertUToPtr",
	"PtrCastToGeneric",
	"GenericCastToPtr",
	"GenericCastToPtrExplicit",
	"Bitcast",
	"#125",
	"SNegate",
	"FNegate",
	"IAdd",
	"FAdd",
	"ISub",
	"FSub",
	"IMul",
	"FMul",
	"UDiv",
	"SDiv",
	"FDiv",
	"UMod",
	"SRem",
	"SMod",
	"FRem",
	"FMod",
	"VectorTimesScalar",
	"MatrixTimesScalar",
	"VectorTimesMatrix",
	"MatrixTimesVector",
	"MatrixTimesMatrix",
	"OuterProduct",
	"Dot",
	"IAddCarry",
	"ISubBorrow",
	"UMulExtended",
	"SMulExtended",
	"#153",
	"Any",
	"All",
	"IsNan",
	"IsInf",
	"IsFinite",
	"IsNormal",
	"SignBitSet",
	"LessOrGreater",
	"Ordered",
	"Unordered",
	"LogicalEqual",
	"LogicalNotEqual",
	"LogicalOr",
	"LogicalAnd",
	"LogicalNot",
	"Select",
	"IEqual",
	"INotEqual",
	"UGreaterThan",
	"SGreaterThan",
	"UGreaterThanEqual",
	"SGreaterThanEqual",
	"ULessThan",
	"SLessThan",
	"ULessThanEqual",
	"SLessThanEqual",
	"FOrdEqual",
	"FUnordEqual",
	"FOrdNotEqual",
	"FUnordNotEqual",
	"FOrdLessThan",
	"FUnordLessThan",
	"FOrdGreaterThan",
	"FUnordGreaterThan",
	"FOrdLessThanEqual",
	"FUnordLessThanEqual",
	"FOrdGreaterThanEqual",
	"FUnordGreaterThanEqual",
	"#192",
	"#193",
	"ShiftRightLogical",
	"ShiftRightArithmetic",
	"ShiftLeftLogical",
	"BitwiseOr",
	"BitwiseXor",
	"BitwiseAnd",
	"Not",
	"BitFieldInsert",
	"BitFieldSExtract",
	"BitFieldUExtract",
	"BitReverse",
	"BitCount",
	"#206",
	"DPdx",
	"DPdy",
	"Fwidth",
	"DPdxFine",
	"DPdyFine",
	"FwidthFine",
	"DPdxCoarse",
	"DPdyCoarse",
	"FwidthCoarse",
	"#216",
	"#217",
	"EmitVertex",
	"EndPrimitive",
	"EmitStreamVertex",
	"EndStreamPrimitive",
	"#222",
	"#223",
	"ControlBarrier",
	"MemoryBarrier",
	"#226",
	"AtomicLoad",
	"AtomicStore",
	"AtomicExchange",
	"AtomicCompareExchange",
	"AtomicCompareExchangeWeak",
	"AtomicIIncrement",
	"AtomicIDecrement",
	"AtomicIAdd",
	"AtomicISub",
	"AtomicSMin",
	"AtomicUMin",
	"AtomicSMax",
	"AtomicUMax",
	"AtomicAnd",
	"AtomicOr",
	"AtomicXor",
	"#243",
	"#244",
	"Phi",
	"LoopMerge",
	"SelectionMerge",
	"Label",
	"Branch",
	"BranchConditional",
	"Switch",
	"Kill",
	"Return",
	"ReturnValue",
	"Unreachable",
	"LifetimeStart",
	"LifetimeStop",
	"#258",
	"GroupAsyncCopy",
	"GroupWaitEvents",
	"GroupAll",
	"GroupAny",
	"GroupBroadcast",
	"GroupIAdd",
	"GroupFAdd",
	"GroupFMin",
	"GroupUMin",
	"GroupSMin",
	"GroupFMax",
	"GroupUMax",
	"GroupSMax",
	"#272",
	"#273",
	"ReadPipe",
	"WritePipe",
	"ReservedReadPipe",
	"ReservedWritePipe",
	"ReserveReadPipePackets",
	"ReserveWritePipePackets",
	"CommitReadPipe",
	"CommitWritePipe",
	"IsValidReserveId",
	"GetNumPipePackets",
	"GetMaxPipePackets",
	"GroupReserveReadPipePackets",
	"GroupReserveWritePipePackets",
	"GroupCommitReadPipe",
	"GroupCommitWritePipe",
	"#289",
	"#290",
	"EnqueueMarker",
	"EnqueueKernel",
	"GetKernelNDrangeSubGroupCount",
	"GetKernelNDrangeMaxSubGroupSize",
	"GetKernelWorkGroupSize",
	"GetKernelPreferredWorkGroupSizeMultiple",
	"RetainEvent",
	"ReleaseEvent",
	"CreateUserEvent",
	"IsValidEvent",
	"SetUserEventStatus",
	"CaptureEventProfilingInfo",
	"GetDefaultQueue",
	"BuildNDRange",
	"ImageSparseSampleImplicitLod",
	"ImageSparseSampleExplicitLod",
	"ImageSparseSampleDrefImplicitLod",
	"ImageSparseSampleDrefExplicitLod",
	"ImageSparseSampleProjImplicitLod",
	"ImageSparseSampleProjExplicitLod",
	"ImageSparseSampleProjDrefImplicitLod",
	"ImageSparseSampleProjDrefExplicitLod",
	"ImageSparseFetch",
	"ImageSparseGather",
	"ImageSparseDrefGather",
	"ImageSparseTexelsResident",
	"NoLine",
	"AtomicFlagTestAndSet",
	"AtomicFlagClear",
	"ImageSparseRead",
	"SizeOf",
	"TypePipeStorage",
	"ConstantPipeStorage",
	"CreatePipeFromPipeStorage",
	"GetKernelLocalSizeForSubgroupCount",
	"GetKernelMaxNumSubgroups",
	"TypeNamedBarrier",
	"NamedBarrierInitialize",
	"MemoryNamedBarrier",
	"ModuleProcessed",
	"ExecutionModeId",
	"DecorateId",
	"GroupNonUniformElect",
	"GroupNonUniformAll",
	"GroupNonUniformAny",
	"GroupNonUniformAllEqual",
	"GroupNonUniformBroadcast",
	"GroupNonUniformBroadcastFirst",
	"GroupNonUniformBallot",
	"GroupNonUniformInverseBallot",
	"GroupNonUniformBallotBitExtract",
	"GroupNonUniformBallotBitCount",
	"GroupNonUniformBallotFindLSB",
	"GroupNonUniformBallotFindMSB",
	"GroupNonUniformShuffle",
	"GroupNonUniformShuffleXor",
	"GroupNonUniformShuffleUp",
	"GroupNonUniformShuffleDown",
	"GroupNonUniformIAdd",
	"GroupNonUniformFAdd",
	"GroupNonUniformIMul",
	"GroupNonUniformFMul",
	"GroupNonUniformSMin",
	"GroupNonUniformUMin",
	"GroupNonUniformFMin",
	"GroupNonUniformSMax",
	"GroupNonUniformUMax",
	"GroupNonUniformFMax",
	"GroupNonUniformBitwiseAnd",
	"GroupNonUniformBitwiseOr",
	"GroupNonUniformBitwiseXor",
	"GroupNonUniformLogicalAnd",
	"GroupNonUniformLogicalOr",
	"GroupNonUniformLogicalXor",
	"GroupNonUniformQuadBroadcast",
	"GroupNonUniformQuadSwap",
};
static_assert(_SMOLV_ARRAY_SIZE(kSpirvOpNames) == kKnownOpsCount, "kSpirvOpNames table mismatch with known SpvOps");


struct OpData
{
	uint8_t hasResult;	// does it have result ID?
	uint8_t hasType;	// does it have type ID?
	uint8_t deltaFromResult; // How many words after (optional) type+result to write out as deltas from result?
	uint8_t varrest;	// should the rest of words be written in varint encoding?
};
static const OpData kSpirvOpData[] =
{
	{0, 0, 0, 0}, // Nop
	{1, 1, 0, 0}, // Undef
	{0, 0, 0, 0}, // SourceContinued
	{0, 0, 0, 1}, // Source
	{0, 0, 0, 0}, // SourceExtension
	{0, 0, 0, 0}, // Name
	{0, 0, 0, 0}, // MemberName
	{0, 0, 0, 0}, // String
	{0, 0, 0, 1}, // Line
	{1, 1, 0, 0}, // #9
	{0, 0, 0, 0}, // Extension
	{1, 0, 0, 0}, // ExtInstImport
	{1, 1, 0, 1}, // ExtInst
	{1, 1, 2, 1}, // VectorShuffleCompact - new in SMOLV
	{0, 0, 0, 1}, // MemoryModel
	{0, 0, 0, 1}, // EntryPoint
	{0, 0, 0, 1}, // ExecutionMode
	{0, 0, 0, 1}, // Capability
	{1, 1, 0, 0}, // #18
	{1, 0, 0, 1}, // TypeVoid
	{1, 0, 0, 1}, // TypeBool
	{1, 0, 0, 1}, // TypeInt
	{1, 0, 0, 1}, // TypeFloat
	{1, 0, 0, 1}, // TypeVector
	{1, 0, 0, 1}, // TypeMatrix
	{1, 0, 0, 1}, // TypeImage
	{1, 0, 0, 1}, // TypeSampler
	{1, 0, 0, 1}, // TypeSampledImage
	{1, 0, 0, 1}, // TypeArray
	{1, 0, 0, 1}, // TypeRuntimeArray
	{1, 0, 0, 1}, // TypeStruct
	{1, 0, 0, 1}, // TypeOpaque
	{1, 0, 0, 1}, // TypePointer
	{1, 0, 0, 1}, // TypeFunction
	{1, 0, 0, 1}, // TypeEvent
	{1, 0, 0, 1}, // TypeDeviceEvent
	{1, 0, 0, 1}, // TypeReserveId
	{1, 0, 0, 1}, // TypeQueue
	{1, 0, 0, 1}, // TypePipe
	{0, 0, 0, 1}, // TypeForwardPointer
	{1, 1, 0, 0}, // #40
	{1, 1, 0, 0}, // ConstantTrue
	{1, 1, 0, 0}, // ConstantFalse
	{1, 1, 0, 0}, // Constant
	{1, 1, 9, 0}, // ConstantComposite
	{1, 1, 0, 1}, // ConstantSampler
	{1, 1, 0, 0}, // ConstantNull
	{1, 1, 0, 0}, // #47
	{1, 1, 0, 0}, // SpecConstantTrue
	{1, 1, 0, 0}, // SpecConstantFalse
	{1, 1, 0, 0}, // SpecConstant
	{1, 1, 9, 0}, // SpecConstantComposite
	{1, 1, 0, 0}, // SpecConstantOp
	{1, 1, 0, 0}, // #53
	{1, 1, 0, 1}, // Function
	{1, 1, 0, 0}, // FunctionParameter
	{0, 0, 0, 0}, // FunctionEnd
	{1, 1, 9, 0}, // FunctionCall
	{1, 1, 0, 0}, // #58
	{1, 1, 0, 1}, // Variable
	{1, 1, 0, 0}, // ImageTexelPointer
	{1, 1, 1, 1}, // Load
	{0, 0, 2, 1}, // Store
	{0, 0, 0, 0}, // CopyMemory
	{0, 0, 0, 0}, // CopyMemorySized
	{1, 1, 0, 1}, // AccessChain
	{1, 1, 0, 0}, // InBoundsAccessChain
	{1, 1, 0, 0}, // PtrAccessChain
	{1, 1, 0, 0}, // ArrayLength
	{1, 1, 0, 0}, // GenericPtrMemSemantics
	{1, 1, 0, 0}, // InBoundsPtrAccessChain
	{0, 0, 0, 1}, // Decorate
	{0, 0, 0, 1}, // MemberDecorate
	{1, 0, 0, 0}, // DecorationGroup
	{0, 0, 0, 0}, // GroupDecorate
	{0, 0, 0, 0}, // GroupMemberDecorate
	{1, 1, 0, 0}, // #76
	{1, 1, 1, 1}, // VectorExtractDynamic
	{1, 1, 2, 1}, // VectorInsertDynamic
	{1, 1, 2, 1}, // VectorShuffle
	{1, 1, 9, 0}, // CompositeConstruct
	{1, 1, 1, 1}, // CompositeExtract
	{1, 1, 2, 1}, // CompositeInsert
	{1, 1, 1, 0}, // CopyObject
	{1, 1, 0, 0}, // Transpose
	{1, 1, 0, 0}, // #85
	{1, 1, 0, 0}, // SampledImage
	{1, 1, 2, 1}, // ImageSampleImplicitLod
	{1, 1, 2, 1}, // ImageSampleExplicitLod
	{1, 1, 3, 1}, // ImageSampleDrefImplicitLod
	{1, 1, 3, 1}, // ImageSampleDrefExplicitLod
	{1, 1, 2, 1}, // ImageSampleProjImplicitLod
	{1, 1, 2, 1}, // ImageSampleProjExplicitLod
	{1, 1, 3, 1}, // ImageSampleProjDrefImplicitLod
	{1, 1, 3, 1}, // ImageSampleProjDrefExplicitLod
	{1, 1, 2, 1}, // ImageFetch
	{1, 1, 3, 1}, // ImageGather
	{1, 1, 3, 1}, // ImageDrefGather
	{1, 1, 2, 1}, // ImageRead
	{0, 0, 3, 1}, // ImageWrite
	{1, 1, 1, 0}, // Image
	{1, 1, 1, 0}, // ImageQueryFormat
	{1, 1, 1, 0}, // ImageQueryOrder
	{1, 1, 2, 0}, // ImageQuerySizeLod
	{1, 1, 1, 0}, // ImageQuerySize
	{1, 1, 2, 0}, // ImageQueryLod
	{1, 1, 1, 0}, // ImageQueryLevels
	{1, 1, 1, 0}, // ImageQuerySamples
	{1, 1, 0, 0}, // #108
	{1, 1, 1, 0}, // ConvertFToU
	{1, 1, 1, 0}, // ConvertFToS
	{1, 1, 1, 0}, // ConvertSToF
	{1, 1, 1, 0}, // ConvertUToF
	{1, 1, 1, 0}, // UConvert
	{1, 1, 1, 0}, // SConvert
	{1, 1, 1, 0}, // FConvert
	{1, 1, 1, 0}, // QuantizeToF16
	{1, 1, 1, 0}, // ConvertPtrToU
	{1, 1, 1, 0}, // SatConvertSToU
	{1, 1, 1, 0}, // SatConvertUToS
	{1, 1, 1, 0}, // ConvertUToPtr
	{1, 1, 1, 0}, // PtrCastToGeneric
	{1, 1, 1, 0}, // GenericCastToPtr
	{1, 1, 1, 1}, // GenericCastToPtrExplicit
	{1, 1, 1, 0}, // Bitcast
	{1, 1, 0, 0}, // #125
	{1, 1, 1, 0}, // SNegate
	{1, 1, 1, 0}, // FNegate
	{1, 1, 2, 0}, // IAdd
	{1, 1, 2, 0}, // FAdd
	{1, 1, 2, 0}, // ISub
	{1, 1, 2, 0}, // FSub
	{1, 1, 2, 0}, // IMul
	{1, 1, 2, 0}, // FMul
	{1, 1, 2, 0}, // UDiv
	{1, 1, 2, 0}, // SDiv
	{1, 1, 2, 0}, // FDiv
	{1, 1, 2, 0}, // UMod
	{1, 1, 2, 0}, // SRem
	{1, 1, 2, 0}, // SMod
	{1, 1, 2, 0}, // FRem
	{1, 1, 2, 0}, // FMod
	{1, 1, 2, 0}, // VectorTimesScalar
	{1, 1, 2, 0}, // MatrixTimesScalar
	{1, 1, 2, 0}, // VectorTimesMatrix
	{1, 1, 2, 0}, // MatrixTimesVector
	{1, 1, 2, 0}, // MatrixTimesMatrix
	{1, 1, 2, 0}, // OuterProduct
	{1, 1, 2, 0}, // Dot
	{1, 1, 2, 0}, // IAddCarry
	{1, 1, 2, 0}, // ISubBorrow
	{1, 1, 2, 0}, // UMulExtended
	{1, 1, 2, 0}, // SMulExtended
	{1, 1, 0, 0}, // #153
	{1, 1, 1, 0}, // Any
	{1, 1, 1, 0}, // All
	{1, 1, 1, 0}, // IsNan
	{1, 1, 1, 0}, // IsInf
	{1, 1, 1, 0}, // IsFinite
	{1, 1, 1, 0}, // IsNormal
	{1, 1, 1, 0}, // SignBitSet
	{1, 1, 2, 0}, // LessOrGreater
	{1, 1, 2, 0}, // Ordered
	{1, 1, 2, 0}, // Unordered
	{1, 1, 2, 0}, // LogicalEqual
	{1, 1, 2, 0}, // LogicalNotEqual
	{1, 1, 2, 0}, // LogicalOr
	{1, 1, 2, 0}, // LogicalAnd
	{1, 1, 1, 0}, // LogicalNot
	{1, 1, 3, 0}, // Select
	{1, 1, 2, 0}, // IEqual
	{1, 1, 2, 0}, // INotEqual
	{1, 1, 2, 0}, // UGreaterThan
	{1, 1, 2, 0}, // SGreaterThan
	{1, 1, 2, 0}, // UGreaterThanEqual
	{1, 1, 2, 0}, // SGreaterThanEqual
	{1, 1, 2, 0}, // ULessThan
	{1, 1, 2, 0}, // SLessThan
	{1, 1, 2, 0}, // ULessThanEqual
	{1, 1, 2, 0}, // SLessThanEqual
	{1, 1, 2, 0}, // FOrdEqual
	{1, 1, 2, 0}, // FUnordEqual
	{1, 1, 2, 0}, // FOrdNotEqual
	{1, 1, 2, 0}, // FUnordNotEqual
	{1, 1, 2, 0}, // FOrdLessThan
	{1, 1, 2, 0}, // FUnordLessThan
	{1, 1, 2, 0}, // FOrdGreaterThan
	{1, 1, 2, 0}, // FUnordGreaterThan
	{1, 1, 2, 0}, // FOrdLessThanEqual
	{1, 1, 2, 0}, // FUnordLessThanEqual
	{1, 1, 2, 0}, // FOrdGreaterThanEqual
	{1, 1, 2, 0}, // FUnordGreaterThanEqual
	{1, 1, 0, 0}, // #192
	{1, 1, 0, 0}, // #193
	{1, 1, 2, 0}, // ShiftRightLogical
	{1, 1, 2, 0}, // ShiftRightArithmetic
	{1, 1, 2, 0}, // ShiftLeftLogical
	{1, 1, 2, 0}, // BitwiseOr
	{1, 1, 2, 0}, // BitwiseXor
	{1, 1, 2, 0}, // BitwiseAnd
	{1, 1, 1, 0}, // Not
	{1, 1, 4, 0}, // BitFieldInsert
	{1, 1, 3, 0}, // BitFieldSExtract
	{1, 1, 3, 0}, // BitFieldUExtract
	{1, 1, 1, 0}, // BitReverse
	{1, 1, 1, 0}, // BitCount
	{1, 1, 0, 0}, // #206
	{1, 1, 0, 0}, // DPdx
	{1, 1, 0, 0}, // DPdy
	{1, 1, 0, 0}, // Fwidth
	{1, 1, 0, 0}, // DPdxFine
	{1, 1, 0, 0}, // DPdyFine
	{1, 1, 0, 0}, // FwidthFine
	{1, 1, 0, 0}, // DPdxCoarse
	{1, 1, 0, 0}, // DPdyCoarse
	{1, 1, 0, 0}, // FwidthCoarse
	{1, 1, 0, 0}, // #216
	{1, 1, 0, 0}, // #217
	{0, 0, 0, 0}, // EmitVertex
	{0, 0, 0, 0}, // EndPrimitive
	{0, 0, 0, 0}, // EmitStreamVertex
	{0, 0, 0, 0}, // EndStreamPrimitive
	{1, 1, 0, 0}, // #222
	{1, 1, 0, 0}, // #223
	{0, 0, 3, 0}, // ControlBarrier
	{0, 0, 2, 0}, // MemoryBarrier
	{1, 1, 0, 0}, // #226
	{1, 1, 0, 0}, // AtomicLoad
	{0, 0, 0, 0}, // AtomicStore
	{1, 1, 0, 0}, // AtomicExchange
	{1, 1, 0, 0}, // AtomicCompareExchange
	{1, 1, 0, 0}, // AtomicCompareExchangeWeak
	{1, 1, 0, 0}, // AtomicIIncrement
	{1, 1, 0, 0}, // AtomicIDecrement
	{1, 1, 0, 0}, // AtomicIAdd
	{1, 1, 0, 0}, // AtomicISub
	{1, 1, 0, 0}, // AtomicSMin
	{1, 1, 0, 0}, // AtomicUMin
	{1, 1, 0, 0}, // AtomicSMax
	{1, 1, 0, 0}, // AtomicUMax
	{1, 1, 0, 0}, // AtomicAnd
	{1, 1, 0, 0}, // AtomicOr
	{1, 1, 0, 0}, // AtomicXor
	{1, 1, 0, 0}, // #243
	{1, 1, 0, 0}, // #244
	{1, 1, 0, 0}, // Phi
	{0, 0, 2, 1}, // LoopMerge
	{0, 0, 1, 1}, // SelectionMerge
	{1, 0, 0, 0}, // Label
	{0, 0, 1, 0}, // Branch
	{0, 0, 3, 1}, // BranchConditional
	{0, 0, 0, 0}, // Switch
	{0, 0, 0, 0}, // Kill
	{0, 0, 0, 0}, // Return
	{0, 0, 0, 0}, // ReturnValue
	{0, 0, 0, 0}, // Unreachable
	{0, 0, 0, 0}, // LifetimeStart
	{0, 0, 0, 0}, // LifetimeStop
	{1, 1, 0, 0}, // #258
	{1, 1, 0, 0}, // GroupAsyncCopy
	{0, 0, 0, 0}, // GroupWaitEvents
	{1, 1, 0, 0}, // GroupAll
	{1, 1, 0, 0}, // GroupAny
	{1, 1, 0, 0}, // GroupBroadcast
	{1, 1, 0, 0}, // GroupIAdd
	{1, 1, 0, 0}, // GroupFAdd
	{1, 1, 0, 0}, // GroupFMin
	{1, 1, 0, 0}, // GroupUMin
	{1, 1, 0, 0}, // GroupSMin
	{1, 1, 0, 0}, // GroupFMax
	{1, 1, 0, 0}, // GroupUMax
	{1, 1, 0, 0}, // GroupSMax
	{1, 1, 0, 0}, // #272
	{1, 1, 0, 0}, // #273
	{1, 1, 0, 0}, // ReadPipe
	{1, 1, 0, 0}, // WritePipe
	{1, 1, 0, 0}, // ReservedReadPipe
	{1, 1, 0, 0}, // ReservedWritePipe
	{1, 1, 0, 0}, // ReserveReadPipePackets
	{1, 1, 0, 0}, // ReserveWritePipePackets
	{0, 0, 0, 0}, // CommitReadPipe
	{0, 0, 0, 0}, // CommitWritePipe
	{1, 1, 0, 0}, // IsValidReserveId
	{1, 1, 0, 0}, // GetNumPipePackets
	{1, 1, 0, 0}, // GetMaxPipePackets
	{1, 1, 0, 0}, // GroupReserveReadPipePackets
	{1, 1, 0, 0}, // GroupReserveWritePipePackets
	{0, 0, 0, 0}, // GroupCommitReadPipe
	{0, 0, 0, 0}, // GroupCommitWritePipe
	{1, 1, 0, 0}, // #289
	{1, 1, 0, 0}, // #290
	{1, 1, 0, 0}, // EnqueueMarker
	{1, 1, 0, 0}, // EnqueueKernel
	{1, 1, 0, 0}, // GetKernelNDrangeSubGroupCount
	{1, 1, 0, 0}, // GetKernelNDrangeMaxSubGroupSize
	{1, 1, 0, 0}, // GetKernelWorkGroupSize
	{1, 1, 0, 0}, // GetKernelPreferredWorkGroupSizeMultiple
	{0, 0, 0, 0}, // RetainEvent
	{0, 0, 0, 0}, // ReleaseEvent
	{1, 1, 0, 0}, // CreateUserEvent
	{1, 1, 0, 0}, // IsValidEvent
	{0, 0, 0, 0}, // SetUserEventStatus
	{0, 0, 0, 0}, // CaptureEventProfilingInfo
	{1, 1, 0, 0}, // GetDefaultQueue
	{1, 1, 0, 0}, // BuildNDRange
	{1, 1, 2, 1}, // ImageSparseSampleImplicitLod
	{1, 1, 2, 1}, // ImageSparseSampleExplicitLod
	{1, 1, 3, 1}, // ImageSparseSampleDrefImplicitLod
	{1, 1, 3, 1}, // ImageSparseSampleDrefExplicitLod
	{1, 1, 2, 1}, // ImageSparseSampleProjImplicitLod
	{1, 1, 2, 1}, // ImageSparseSampleProjExplicitLod
	{1, 1, 3, 1}, // ImageSparseSampleProjDrefImplicitLod
	{1, 1, 3, 1}, // ImageSparseSampleProjDrefExplicitLod
	{1, 1, 2, 1}, // ImageSparseFetch
	{1, 1, 3, 1}, // ImageSparseGather
	{1, 1, 3, 1}, // ImageSparseDrefGather
	{1, 1, 1, 0}, // ImageSparseTexelsResident
	{0, 0, 0, 0}, // NoLine
	{1, 1, 0, 0}, // AtomicFlagTestAndSet
	{0, 0, 0, 0}, // AtomicFlagClear
	{1, 1, 0, 0}, // ImageSparseRead
	{1, 1, 0, 0}, // SizeOf
	{1, 1, 0, 0}, // TypePipeStorage
	{1, 1, 0, 0}, // ConstantPipeStorage
	{1, 1, 0, 0}, // CreatePipeFromPipeStorage
	{1, 1, 0, 0}, // GetKernelLocalSizeForSubgroupCount
	{1, 1, 0, 0}, // GetKernelMaxNumSubgroups
	{1, 1, 0, 0}, // TypeNamedBarrier
	{1, 1, 0, 1}, // NamedBarrierInitialize
	{0, 0, 2, 1}, // MemoryNamedBarrier
	{1, 1, 0, 0}, // ModuleProcessed
	{0, 0, 0, 1}, // ExecutionModeId
	{0, 0, 0, 1}, // DecorateId
	{1, 1, 1, 1}, // GroupNonUniformElect
	{1, 1, 1, 1}, // GroupNonUniformAll
	{1, 1, 1, 1}, // GroupNonUniformAny
	{1, 1, 1, 1}, // GroupNonUniformAllEqual
	{1, 1, 1, 1}, // GroupNonUniformBroadcast
	{1, 1, 1, 1}, // GroupNonUniformBroadcastFirst
	{1, 1, 1, 1}, // GroupNonUniformBallot
	{1, 1, 1, 1}, // GroupNonUniformInverseBallot
	{1, 1, 1, 1}, // GroupNonUniformBallotBitExtract
	{1, 1, 1, 1}, // GroupNonUniformBallotBitCount
	{1, 1, 1, 1}, // GroupNonUniformBallotFindLSB
	{1, 1, 1, 1}, // GroupNonUniformBallotFindMSB
	{1, 1, 1, 1}, // GroupNonUniformShuffle
	{1, 1, 1, 1}, // GroupNonUniformShuffleXor
	{1, 1, 1, 1}, // GroupNonUniformShuffleUp
	{1, 1, 1, 1}, // GroupNonUniformShuffleDown
	{1, 1, 1, 1}, // GroupNonUniformIAdd
	{1, 1, 1, 1}, // GroupNonUniformFAdd
	{1, 1, 1, 1}, // GroupNonUniformIMul
	{1, 1, 1, 1}, // GroupNonUniformFMul
	{1, 1, 1, 1}, // GroupNonUniformSMin
	{1, 1, 1, 1}, // GroupNonUniformUMin
	{1, 1, 1, 1}, // GroupNonUniformFMin
	{1, 1, 1, 1}, // GroupNonUniformSMax
	{1, 1, 1, 1}, // GroupNonUniformUMax
	{1, 1, 1, 1}, // GroupNonUniformFMax
	{1, 1, 1, 1}, // GroupNonUniformBitwiseAnd
	{1, 1, 1, 1}, // GroupNonUniformBitwiseOr
	{1, 1, 1, 1}, // GroupNonUniformBitwiseXor
	{1, 1, 1, 1}, // GroupNonUniformLogicalAnd
	{1, 1, 1, 1}, // GroupNonUniformLogicalOr
	{1, 1, 1, 1}, // GroupNonUniformLogicalXor
	{1, 1, 1, 1}, // GroupNonUniformQuadBroadcast
	{1, 1, 1, 1}, // GroupNonUniformQuadSwap
};
static_assert(_SMOLV_ARRAY_SIZE(kSpirvOpData) == kKnownOpsCount, "kSpirvOpData table mismatch with known SpvOps");

// Instruction encoding depends on the table that describes the various SPIR-V opcodes.
// Whenever we change or expand the table, we need to bump up the SMOL-V version, and make
// sure that we can still decode files encoded by an older version.
static int smolv_GetKnownOpsCount(int version)
{
	if (version == 0)
		return SpvOpModuleProcessed+1;
	if (version == 1) // 2020 February, version 1 added ExecutionModeId..GroupNonUniformQuadSwap
		return SpvOpGroupNonUniformQuadSwap+1;
	return 0;
}

static bool smolv_OpHasResult(SpvOp op, int opsCount)
{
	if (op < 0 || op >= opsCount)
		return false;
	return kSpirvOpData[op].hasResult != 0;
}

static bool smolv_OpHasType(SpvOp op, int opsCount)
{
	if (op < 0 || op >= opsCount)
		return false;
	return kSpirvOpData[op].hasType != 0;
}

static int smolv_OpDeltaFromResult(SpvOp op, int opsCount)
{
	if (op < 0 || op >= opsCount)
		return 0;
	return kSpirvOpData[op].deltaFromResult;
}

static bool smolv_OpVarRest(SpvOp op, int opsCount)
{
	if (op < 0 || op >= opsCount)
		return false;
	return kSpirvOpData[op].varrest != 0;
}

static bool smolv_OpDebugInfo(SpvOp op, int opsCount)
{
	return
		op == SpvOpSourceContinued ||
		op == SpvOpSource ||
		op == SpvOpSourceExtension ||
		op == SpvOpName ||
		op == SpvOpMemberName ||
		op == SpvOpString ||
		op == SpvOpLine ||
		op == SpvOpNoLine ||
		op == SpvOpModuleProcessed;
}


static int smolv_DecorationExtraOps(int dec)
{
	if (dec == 0 || (dec >= 2 && dec <= 5)) // RelaxedPrecision, Block..ColMajor
		return 0;
	if (dec >= 29 && dec <= 37) // Stream..XfbStride
		return 1;
	return -1; // unknown, encode length
}


// --------------------------------------------------------------------------------------------


static bool smolv_CheckGenericHeader(const uint32_t* words, size_t wordCount, uint32_t expectedMagic, uint32_t versionMask)
{
	if (!words)
		return false;
	if (wordCount < 5)
		return false;
	
	uint32_t headerMagic = words[0];
	if (headerMagic != expectedMagic)
		return false;
	uint32_t headerVersion = words[1] & versionMask;
	if (headerVersion < 0x00010000 || headerVersion > 0x00010500)
		return false; // only support 1.0 through 1.5
	
	return true;
}

static const int kSpirVHeaderMagic = 0x07230203;
static const int kSmolHeaderMagic = 0x534D4F4C; // "SMOL"

static const int kSmolCurrEncodingVersion = 1;

static bool smolv_CheckSpirVHeader(const uint32_t* words, size_t wordCount)
{
	//@TODO: if SPIR-V header magic was reversed, that means the file got written
	// in a "big endian" order. Need to byteswap all words then.
	return smolv_CheckGenericHeader(words, wordCount, kSpirVHeaderMagic, 0xFFFFFFFF);
}
static bool smolv_CheckSmolHeader(const uint8_t* bytes, size_t byteCount)
{
	if (!smolv_CheckGenericHeader((const uint32_t*)bytes, byteCount/4, kSmolHeaderMagic, 0x00FFFFFF))
		return false;
	if (byteCount < 24) // one more word past header to store decoded length
		return false;
	// SMOL-V version
	int smolVersion = ((const uint32_t*)bytes)[1] >> 24;
	if (smolVersion < 0 || smolVersion > kSmolCurrEncodingVersion)
		return false;
	return true;
}


static void smolv_Write4(smolv::ByteArray& arr, uint32_t v)
{
	arr.push_back(v & 0xFF);
	arr.push_back((v >> 8) & 0xFF);
	arr.push_back((v >> 16) & 0xFF);
	arr.push_back(v >> 24);
}

static void smolv_Write4(uint8_t*& buf, uint32_t v)
{
	memcpy(buf, &v, 4);
	buf += 4;
}


static bool smolv_Read4(const uint8_t*& data, const uint8_t* dataEnd, uint32_t& outv)
{
	if (data + 4 > dataEnd)
		return false;
	outv = (data[0]) | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
	data += 4;
	return true;
}


// --------------------------------------------------------------------------------------------

// Variable-length integer encoding for unsigned integers. In each byte:
// - highest bit set if more bytes follow, cleared if this is last byte.
// - other 7 bits are the actual value payload.
// Takes 1-5 bytes to encode an integer (values between 0 and 127 take one byte, etc.).

static void smolv_WriteVarint(smolv::ByteArray& arr, uint32_t v)
{
	while (v > 127)
	{
		arr.push_back((v & 127) | 128);
		v >>= 7;
	}
	arr.push_back(v & 127);
}

static bool smolv_ReadVarint(const uint8_t*& data, const uint8_t* dataEnd, uint32_t& outVal)
{
	uint32_t v = 0;
	uint32_t shift = 0;
	while (data < dataEnd)
	{
		uint8_t b = *data;
		v |= (b & 127) << shift;
		shift += 7;
		data++;
		if (!(b & 128))
			break;
	}
	outVal = v;
	return true; //@TODO: report failures
}

static uint32_t smolv_ZigEncode(int32_t i)
{
	return (uint32_t(i) << 1) ^ (i >> 31);
}

static int32_t smolv_ZigDecode(uint32_t u)
{
	 return (u & 1) ? ((u >> 1) ^ ~0) : (u >> 1);
}


// Remap most common Op codes (Load, Store, Decorate, VectorShuffle etc.) to be in < 16 range, for
// more compact varint encoding. This basically swaps rarely used op values that are < 16 with the
// ones that are common.

static SpvOp smolv_RemapOp(SpvOp op)
{
#	define _SMOLV_SWAP_OP(op1,op2) if (op==op1) return op2; if (op==op2) return op1
	_SMOLV_SWAP_OP(SpvOpDecorate,SpvOpNop); // 0: 24%
	_SMOLV_SWAP_OP(SpvOpLoad,SpvOpUndef); // 1: 17%
	_SMOLV_SWAP_OP(SpvOpStore,SpvOpSourceContinued); // 2: 9%
	_SMOLV_SWAP_OP(SpvOpAccessChain,SpvOpSource); // 3: 7.2%
	_SMOLV_SWAP_OP(SpvOpVectorShuffle,SpvOpSourceExtension); // 4: 5.0%
	// Name - already small enum value - 5: 4.4%
	// MemberName - already small enum value - 6: 2.9%
	_SMOLV_SWAP_OP(SpvOpMemberDecorate,SpvOpString); // 7: 4.0%
	_SMOLV_SWAP_OP(SpvOpLabel,SpvOpLine); // 8: 0.9%
	_SMOLV_SWAP_OP(SpvOpVariable,(SpvOp)9); // 9: 3.9%
	_SMOLV_SWAP_OP(SpvOpFMul,SpvOpExtension); // 10: 3.9%
	_SMOLV_SWAP_OP(SpvOpFAdd,SpvOpExtInstImport); // 11: 2.5%
	// ExtInst - already small enum value - 12: 1.2%
	// VectorShuffleCompact - already small enum value - used for compact shuffle encoding
	_SMOLV_SWAP_OP(SpvOpTypePointer,SpvOpMemoryModel); // 14: 2.2%
	_SMOLV_SWAP_OP(SpvOpFNegate,SpvOpEntryPoint); // 15: 1.1%
#	undef _SMOLV_SWAP_OP
	return op;
}


// For most compact varint encoding of common instructions, the instruction length should come out
// into 3 bits (be <8). SPIR-V instruction lengths are always at least 1, and for some other
// instructions they are guaranteed to be some other minimum length. Adjust the length before encoding,
// and after decoding accordingly.

static uint32_t smolv_EncodeLen(SpvOp op, uint32_t len)
{
	len--;
	if (op == SpvOpVectorShuffle)			len -= 4;
	if (op == SpvOpVectorShuffleCompact)	len -= 4;
	if (op == SpvOpDecorate)				len -= 2;
	if (op == SpvOpLoad)					len -= 3;
	if (op == SpvOpAccessChain)				len -= 3;
	return len;
}

static uint32_t smolv_DecodeLen(SpvOp op, uint32_t len)
{
	len++;
	if (op == SpvOpVectorShuffle)			len += 4;
	if (op == SpvOpVectorShuffleCompact)	len += 4;
	if (op == SpvOpDecorate)				len += 2;
	if (op == SpvOpLoad)					len += 3;
	if (op == SpvOpAccessChain)				len += 3;
	return len;
}


// Shuffling bits of length + opcode to be more compact in varint encoding in typical cases:
// 0x LLLL OOOO is how SPIR-V encodes it (L=length, O=op), we shuffle into:
// 0x LLLO OOLO, so that common case (op<16, len<8) is encoded into one byte.

static bool smolv_WriteLengthOp(smolv::ByteArray& arr, uint32_t len, SpvOp op)
{
	len = smolv_EncodeLen(op, len);
	// SPIR-V length field is 16 bits; if we get a larger value that means something
	// was wrong, e.g. a vector shuffle instruction with less than 4 words (and our
	// adjustment to common lengths in smolv_EncodeLen wrapped around)
	if (len > 0xFFFF)
		return false;
	op = smolv_RemapOp(op);
	uint32_t oplen = ((len >> 4) << 20) | ((op >> 4) << 8) | ((len & 0xF) << 4) | (op & 0xF);
	smolv_WriteVarint(arr, oplen);
	return true;
}

static bool smolv_ReadLengthOp(const uint8_t*& data, const uint8_t* dataEnd, uint32_t& outLen, SpvOp& outOp)
{
	uint32_t val;
	if (!smolv_ReadVarint(data, dataEnd, val))
		return false;
	outLen = ((val >> 20) << 4) | ((val >> 4) & 0xF);
	outOp = (SpvOp)(((val >> 4) & 0xFFF0) | (val & 0xF));

	outOp = smolv_RemapOp(outOp);
	outLen = smolv_DecodeLen(outOp, outLen);
	return true;
}



#define _SMOLV_READ_OP(len, words, op) \
	uint32_t len = words[0] >> 16; \
	if (len < 1) return false; /* malformed instruction, length needs to be at least 1 */ \
	if (words + len > wordsEnd) return false; /* malformed instruction, goes past end of data */ \
	SpvOp op = (SpvOp)(words[0] & 0xFFFF)


bool smolv::Encode(const void* spirvData, size_t spirvSize, ByteArray& outSmolv, uint32_t flags, StripOpNameFilterFunc stripFilter)
{
	const size_t wordCount = spirvSize / 4;
	if (wordCount * 4 != spirvSize)
		return false;
	const uint32_t* words = (const uint32_t*)spirvData;
	const uint32_t* wordsEnd = words + wordCount;
	if (!smolv_CheckSpirVHeader(words, wordCount))
		return false;

	// reserve space in output (typical compression is to about 30%; reserve half of input space)
	outSmolv.reserve(outSmolv.size() + spirvSize/2);

	// header (matches SPIR-V one, except different magic)
	smolv_Write4(outSmolv, kSmolHeaderMagic);
	smolv_Write4(outSmolv, (words[1] & 0x00FFFFFF) + (kSmolCurrEncodingVersion<<24)); // SPIR-V version (_XXX) + SMOL-V version (X___)
	smolv_Write4(outSmolv, words[2]); // generator
	smolv_Write4(outSmolv, words[3]); // bound
	smolv_Write4(outSmolv, words[4]); // schema

	const size_t headerSpirvSizeOffset = outSmolv.size(); // size field may get updated later if stripping is enabled
	smolv_Write4(outSmolv, (uint32_t)spirvSize); // space needed to decode (i.e. original SPIR-V size)

	size_t strippedSpirvWordCount = wordCount;
	uint32_t prevResult = 0;
	uint32_t prevDecorate = 0;
	
	const int knownOpsCount = smolv_GetKnownOpsCount(kSmolCurrEncodingVersion);

	words += 5;
	while (words < wordsEnd)
	{
		_SMOLV_READ_OP(instrLen, words, op);

		if ((flags & kEncodeFlagStripDebugInfo) && smolv_OpDebugInfo(op, knownOpsCount))
		{
			if (!stripFilter || op != SpvOpName || !stripFilter(reinterpret_cast<const char*>(&words[2])))
			{
				strippedSpirvWordCount -= instrLen;
				words += instrLen;
				continue;
			}
		}

		// A usual case of vector shuffle, with less than 4 components, each with a value
		// in [0..3] range: encode it in a more compact form, with the swizzle pattern in one byte.
		// Turn this into a VectorShuffleCompact instruction, that takes up unused slot in Ops.
		uint32_t swizzle = 0;
		if (op == SpvOpVectorShuffle && instrLen <= 9)
		{
			uint32_t swz0 = instrLen > 5 ? words[5] : 0;
			uint32_t swz1 = instrLen > 6 ? words[6] : 0;
			uint32_t swz2 = instrLen > 7 ? words[7] : 0;
			uint32_t swz3 = instrLen > 8 ? words[8] : 0;
			if (swz0 < 4 && swz1 < 4 && swz2 < 4 && swz3 < 4)
			{
				op = SpvOpVectorShuffleCompact;
				swizzle = (swz0 << 6) | (swz1 << 4) | (swz2 << 2) | (swz3);
			}
		}

		// length + opcode
		if (!smolv_WriteLengthOp(outSmolv, instrLen, op))
			return false;

		size_t ioffs = 1;
		// write type as varint, if we have it
		if (smolv_OpHasType(op, knownOpsCount))
		{
			if (ioffs >= instrLen)
				return false;
			smolv_WriteVarint(outSmolv, words[ioffs]);
			ioffs++;
		}
		// write result as delta+zig+varint, if we have it
		if (smolv_OpHasResult(op, knownOpsCount))
		{
			if (ioffs >= instrLen)
				return false;
			uint32_t v = words[ioffs];
			smolv_WriteVarint(outSmolv, smolv_ZigEncode(v - prevResult)); // some deltas are negative, use zig
			prevResult = v;
			ioffs++;
		}

		// Decorate & MemberDecorate: IDs relative to previous decorate
		if (op == SpvOpDecorate || op == SpvOpMemberDecorate)
		{
			if (ioffs >= instrLen)
				return false;
			uint32_t v = words[ioffs];
			smolv_WriteVarint(outSmolv, smolv_ZigEncode(v - prevDecorate)); // spirv-remapped deltas often negative, use zig
			prevDecorate = v;
			ioffs++;
		}

		// MemberDecorate special encoding: whole row of MemberDecorate instructions is often referring
		// to the same type and linearly increasing member indices. Scan ahead to see how many we have,
		// and encode whole bunch as one.
		if (op == SpvOpMemberDecorate)
		{
			// scan ahead until we reach end, non-member-decoration or different type
			const uint32_t decorationType = words[ioffs-1];
			const uint32_t* memberWords = words;
			uint32_t prevIndex = 0;
			uint32_t prevOffset = 0;
			// write a byte on how many we have encoded as a bunch
			size_t countLocation = outSmolv.size();
			outSmolv.push_back(0);
			int count = 0;
			while (memberWords < wordsEnd && count < 255)
			{
				_SMOLV_READ_OP(memberLen, memberWords, memberOp);
				if (memberOp != SpvOpMemberDecorate)
					break;
				if (memberLen < 4)
					return false; // invalid input
				if (memberWords[1] != decorationType)
					break;

				// write member index as delta from previous
				uint32_t memberIndex = memberWords[2];
				smolv_WriteVarint(outSmolv, memberIndex - prevIndex);
				prevIndex = memberIndex;

				// decoration (and length if not common/known)
				uint32_t memberDec = memberWords[3];
				smolv_WriteVarint(outSmolv, memberDec);
				const int knownExtraOps = smolv_DecorationExtraOps(memberDec);
				if (knownExtraOps == -1)
					smolv_WriteVarint(outSmolv, memberLen-4);
				else if (unsigned(knownExtraOps) + 4 != memberLen)
					return false; // invalid input

				// Offset decorations are most often linearly increasing, so encode as deltas
				if (memberDec == 35) // Offset
				{
					if (memberLen != 5)
						return false;
					smolv_WriteVarint(outSmolv, memberWords[4]-prevOffset);
					prevOffset = memberWords[4];
				}
				else
				{
					// write rest of decorations as varint
					for (uint32_t i = 4; i < memberLen; ++i)
						smolv_WriteVarint(outSmolv, memberWords[i]);
				}

				memberWords += memberLen;
				++count;
			}
			outSmolv[countLocation] = uint8_t(count);
			words = memberWords;
			continue;
		}

		// Write out this many IDs, encoding them relative+zigzag to result ID
		int relativeCount = smolv_OpDeltaFromResult(op, knownOpsCount);
		for (int i = 0; i < relativeCount && ioffs < instrLen; ++i, ++ioffs)
		{
			if (ioffs >= instrLen)
				return false;
			uint32_t delta = prevResult - words[ioffs];
			// some deltas are negative (often on branches, or if program was processed by spirv-remap),
			// so use zig encoding
			smolv_WriteVarint(outSmolv, smolv_ZigEncode(delta));
		}

		if (op == SpvOpVectorShuffleCompact)
		{
			// compact vector shuffle, just write out single swizzle byte
			outSmolv.push_back(uint8_t(swizzle));
			ioffs = instrLen;
		}
		else if (smolv_OpVarRest(op, knownOpsCount))
		{
			// write out rest of words with variable encoding (expected to be small integers)
			for (; ioffs < instrLen; ++ioffs)
				smolv_WriteVarint(outSmolv, words[ioffs]);
		}
		else
		{
			// write out rest of words without any encoding
			for (; ioffs < instrLen; ++ioffs)
				smolv_Write4(outSmolv, words[ioffs]);
		}
		
		words += instrLen;
	}

	if (strippedSpirvWordCount != wordCount)
	{
		uint8_t* headerSpirvSize = &outSmolv[headerSpirvSizeOffset];
		smolv_Write4(headerSpirvSize, (uint32_t)strippedSpirvWordCount * 4);
	}
	
	return true;
}


size_t smolv::GetDecodedBufferSize(const void* smolvData, size_t smolvSize)
{
	if (!smolv_CheckSmolHeader((const uint8_t*)smolvData, smolvSize))
		return 0;
	const uint32_t* words = (const uint32_t*)smolvData;
	return words[5];
}


bool smolv::Decode(const void* smolvData, size_t smolvSize, void* spirvOutputBuffer, size_t spirvOutputBufferSize, uint32_t flags)
{
	// check header, and whether we have enough output buffer space
	const size_t neededBufferSize = GetDecodedBufferSize(smolvData, smolvSize);
	if (neededBufferSize == 0)
		return false; // invalid SMOL-V
	if (spirvOutputBufferSize < neededBufferSize)
		return false; // not enough space in output buffer
	if (spirvOutputBuffer == NULL)
		return false; // output buffer is null

	const uint8_t* bytes = (const uint8_t*)smolvData;
	const uint8_t* bytesEnd = bytes + smolvSize;

	uint8_t* outSpirv = (uint8_t*)spirvOutputBuffer;
	
	uint32_t val;
	int smolVersion = 0;

	// header
	smolv_Write4(outSpirv, kSpirVHeaderMagic); bytes += 4;
	smolv_Read4(bytes, bytesEnd, val); smolVersion = val >> 24; val &= 0x00FFFFFF; smolv_Write4(outSpirv, val); // version
	smolv_Read4(bytes, bytesEnd, val); smolv_Write4(outSpirv, val); // generator
	smolv_Read4(bytes, bytesEnd, val); smolv_Write4(outSpirv, val); // bound
	smolv_Read4(bytes, bytesEnd, val); smolv_Write4(outSpirv, val); // schema
	bytes += 4; // decode buffer size
	
	// there are two SMOL-V encoding versions, both not indicating anything in their header version field:
	// one that is called "before zero" here (2016-08-31 code). Support decoding that one only by presence
	// of this special flag.
	const bool beforeZeroVersion = smolVersion == 0 && (flags & kDecodeFlagUse20160831AsZeroVersion) != 0;

	const int knownOpsCount = smolv_GetKnownOpsCount(smolVersion);

	uint32_t prevResult = 0;
	uint32_t prevDecorate = 0;

	while (bytes < bytesEnd)
	{
		// read length + opcode
		uint32_t instrLen;
		SpvOp op;
		if (!smolv_ReadLengthOp(bytes, bytesEnd, instrLen, op))
			return false;
		const bool wasSwizzle = (op == SpvOpVectorShuffleCompact);
		if (wasSwizzle)
			op = SpvOpVectorShuffle;
		smolv_Write4(outSpirv, (instrLen << 16) | op);

		size_t ioffs = 1;

		// read type as varint, if we have it
		if (smolv_OpHasType(op, knownOpsCount))
		{
			if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
			smolv_Write4(outSpirv, val);
			ioffs++;
		}
		// read result as delta+varint, if we have it
		if (smolv_OpHasResult(op, knownOpsCount))
		{
			if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
			val = prevResult + smolv_ZigDecode(val);
			smolv_Write4(outSpirv, val);
			prevResult = val;
			ioffs++;
		}
		
		// Decorate: IDs relative to previous decorate
		if (op == SpvOpDecorate || op == SpvOpMemberDecorate)
		{
			if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
			// "before zero" version did not use zig encoding for the value
			val = prevDecorate + (beforeZeroVersion ? val : smolv_ZigDecode(val));
			smolv_Write4(outSpirv, val);
			prevDecorate = val;
			ioffs++;
		}

		// MemberDecorate special decoding
		if (op == SpvOpMemberDecorate && !beforeZeroVersion)
		{
			if (bytes >= bytesEnd)
				return false; // broken input
			int count = *bytes++;
			int prevIndex = 0;
			int prevOffset = 0;
			for (int m = 0; m < count; ++m)
			{
				// read member index
				uint32_t memberIndex;
				if (!smolv_ReadVarint(bytes, bytesEnd, memberIndex)) return false;
				memberIndex += prevIndex;
				prevIndex = memberIndex;
				
				// decoration (and length if not common/known)
				uint32_t memberDec;
				if (!smolv_ReadVarint(bytes, bytesEnd, memberDec)) return false;
				const int knownExtraOps = smolv_DecorationExtraOps(memberDec);
				uint32_t memberLen;
				if (knownExtraOps == -1)
				{
					if (!smolv_ReadVarint(bytes, bytesEnd, memberLen)) return false;
					memberLen += 4;
				}
				else
					memberLen = 4 + knownExtraOps;

				// write SPIR-V op+length (unless it's first member decoration, in which case it was written before)
				if (m != 0)
				{
					smolv_Write4(outSpirv, (memberLen << 16) | op);
					smolv_Write4(outSpirv, prevDecorate);
				}
				smolv_Write4(outSpirv, memberIndex);
				smolv_Write4(outSpirv, memberDec);
				// Special case for Offset decorations
				if (memberDec == 35) // Offset
				{
					if (memberLen != 5)
						return false;
					if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
					val += prevOffset;
					smolv_Write4(outSpirv, val);
					prevOffset = val;
				}
				else
				{
					for (uint32_t i = 4; i < memberLen; ++i)
					{
						if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
						smolv_Write4(outSpirv, val);
					}
				}
			}
			continue;
		}

		// Read this many IDs, that are relative to result ID
		int relativeCount = smolv_OpDeltaFromResult(op, knownOpsCount);
		// "before zero" version only used zig encoding for IDs of several ops; after
		// that ops got zig encoding for their IDs
		bool zigDecodeVals = true;
		if (beforeZeroVersion)
		{
			if (op != SpvOpControlBarrier && op != SpvOpMemoryBarrier && op != SpvOpLoopMerge && op != SpvOpSelectionMerge && op != SpvOpBranch && op != SpvOpBranchConditional && op != SpvOpMemoryNamedBarrier)
				zigDecodeVals = false;
		}
		for (int i = 0; i < relativeCount && ioffs < instrLen; ++i, ++ioffs)
		{
			if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
			if (zigDecodeVals)
				val = smolv_ZigDecode(val);
			smolv_Write4(outSpirv, prevResult - val);
		}

		if (wasSwizzle && instrLen <= 9)
		{
			uint32_t swizzle = *bytes++;
			if (instrLen > 5) smolv_Write4(outSpirv, (swizzle >> 6) & 3);
			if (instrLen > 6) smolv_Write4(outSpirv, (swizzle >> 4) & 3);
			if (instrLen > 7) smolv_Write4(outSpirv, (swizzle >> 2) & 3);
			if (instrLen > 8) smolv_Write4(outSpirv, swizzle & 3);
		}
		else if (smolv_OpVarRest(op, knownOpsCount))
		{
			// read rest of words with variable encoding
			for (; ioffs < instrLen; ++ioffs)
			{
				if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
				smolv_Write4(outSpirv, val);
			}
		}
		else
		{
			// read rest of words without any encoding
			for (; ioffs < instrLen; ++ioffs)
			{
				if (!smolv_Read4(bytes, bytesEnd, val)) return false;
				smolv_Write4(outSpirv, val);
			}
		}
	}

	if ((uint8_t*)spirvOutputBuffer + neededBufferSize != outSpirv)
		return false; // something went wrong during decoding? we should have decoded to exact output size
	
	return true;
}



// --------------------------------------------------------------------------------------------
// Calculating instruction count / space stats on SPIR-V and SMOL-V


struct smolv::Stats
{
	Stats() { memset(this, 0, sizeof(*this)); }
	size_t opCounts[kKnownOpsCount];
	size_t opSizes[kKnownOpsCount];
	size_t smolOpSizes[kKnownOpsCount];
	size_t varintCountsOp[6];
	size_t varintCountsType[6];
	size_t varintCountsRes[6];
	size_t varintCountsOther[6];
	size_t totalOps;
	size_t totalSize;
	size_t totalSizeSmol;
	size_t inputCount;
};


smolv::Stats* smolv::StatsCreate()
{
	return new Stats();
}

void smolv::StatsDelete(smolv::Stats *s)
{
	delete s;
}


bool smolv::StatsCalculate(smolv::Stats* stats, const void* spirvData, size_t spirvSize)
{
	if (!stats)
		return false;

	const size_t wordCount = spirvSize / 4;
	if (wordCount * 4 != spirvSize)
		return false;
	const uint32_t* words = (const uint32_t*)spirvData;
	const uint32_t* wordsEnd = words + wordCount;
	if (!smolv_CheckSpirVHeader(words, wordCount))
		return false;
	words += 5;
	
	stats->inputCount++;
	stats->totalSize += wordCount;

	while (words < wordsEnd)
	{
		_SMOLV_READ_OP(instrLen, words, op);

		if (op < kKnownOpsCount)
		{
			stats->opCounts[op]++;
			stats->opSizes[op] += instrLen;
		}
		words += instrLen;
		stats->totalOps++;
	}
	
	return true;
}


bool smolv::StatsCalculateSmol(smolv::Stats* stats, const void* smolvData, size_t smolvSize)
{
	if (!stats)
		return false;

	// debugging helper to dump all encoded bytes to stdout, keep at "if 0"
#	if 0
#		define _SMOLV_DEBUG_PRINT_ENCODED_BYTES() { \
			printf("Op %-22s ", op < kKnownOpsCount ? kSpirvOpNames[op] : "???"); \
			for (const uint8_t* b = instrBegin; b < bytes; ++b) \
				printf("%02x ", *b); \
			printf("\n"); \
		}
#	else
#		define _SMOLV_DEBUG_PRINT_ENCODED_BYTES() {}
#	endif
	
	const uint8_t* bytes = (const uint8_t*)smolvData;
	const uint8_t* bytesEnd = bytes + smolvSize;
	if (!smolv_CheckSmolHeader(bytes, smolvSize))
		return false;

	uint32_t val;
	int smolVersion;
	bytes += 4;
	smolv_Read4(bytes, bytesEnd, val); smolVersion = val >> 24;
	const int knownOpsCount = smolv_GetKnownOpsCount(smolVersion);
	bytes += 16;
	
	stats->totalSizeSmol += smolvSize;
	
	while (bytes < bytesEnd)
	{
		const uint8_t* instrBegin = bytes;
		const uint8_t* varBegin;

		// read length + opcode
		uint32_t instrLen;
		SpvOp op;
		varBegin = bytes;
		if (!smolv_ReadLengthOp(bytes, bytesEnd, instrLen, op))
			return false;
		const bool wasSwizzle = (op == SpvOpVectorShuffleCompact);
		if (wasSwizzle)
			op = SpvOpVectorShuffle;
		stats->varintCountsOp[bytes-varBegin]++;
		
		size_t ioffs = 1;
		if (smolv_OpHasType(op, knownOpsCount))
		{
			varBegin = bytes;
			if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
			stats->varintCountsType[bytes-varBegin]++;
			ioffs++;
		}
		if (smolv_OpHasResult(op, knownOpsCount))
		{
			varBegin = bytes;
			if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
			stats->varintCountsRes[bytes-varBegin]++;
			ioffs++;
		}
		
		if (op == SpvOpDecorate || op == SpvOpMemberDecorate)
		{
			if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
			ioffs++;
		}
		// MemberDecorate special decoding
		if (op == SpvOpMemberDecorate)
		{
			if (bytes >= bytesEnd)
				return false; // broken input
			int count = *bytes++;
			for (int m = 0; m < count; ++m)
			{
				uint32_t memberIndex;
				if (!smolv_ReadVarint(bytes, bytesEnd, memberIndex)) return false;
				uint32_t memberDec;
				if (!smolv_ReadVarint(bytes, bytesEnd, memberDec)) return false;
				const int knownExtraOps = smolv_DecorationExtraOps(memberDec);
				uint32_t memberLen;
				if (knownExtraOps == -1)
				{
					if (!smolv_ReadVarint(bytes, bytesEnd, memberLen)) return false;
					memberLen += 4;
				}
				else
					memberLen = 4 + knownExtraOps;
				for (uint32_t i = 4; i < memberLen; ++i)
				{
					if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
				}
			}
			stats->smolOpSizes[op] += bytes - instrBegin;
			_SMOLV_DEBUG_PRINT_ENCODED_BYTES();
			continue;
		}

		int relativeCount = smolv_OpDeltaFromResult(op, knownOpsCount);
		for (int i = 0; i < relativeCount && ioffs < instrLen; ++i, ++ioffs)
		{
			varBegin = bytes;
			if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
			stats->varintCountsRes[bytes-varBegin]++;
		}

		if (wasSwizzle && instrLen <= 9)
		{
			bytes++;
		}
		else if (smolv_OpVarRest(op, knownOpsCount))
		{
			for (; ioffs < instrLen; ++ioffs)
			{
				varBegin = bytes;
				if (!smolv_ReadVarint(bytes, bytesEnd, val)) return false;
				stats->varintCountsOther[bytes-varBegin]++;
			}
		}
		else
		{
			for (; ioffs < instrLen; ++ioffs)
			{
				if (!smolv_Read4(bytes, bytesEnd, val)) return false;
			}
		}
		
		if (op < kKnownOpsCount)
		{
			stats->smolOpSizes[op] += bytes - instrBegin;
		}
		_SMOLV_DEBUG_PRINT_ENCODED_BYTES();
	}
	
	return true;
}

static bool CompareOpCounters (std::pair<SpvOp,size_t> a, std::pair<SpvOp,size_t> b)
{
	return a.second > b.second;
}

void smolv::StatsPrint(const Stats* stats)
{
	if (!stats)
		return;

	typedef std::pair<SpvOp,size_t> OpCounter;
	OpCounter counts[kKnownOpsCount];
	OpCounter sizes[kKnownOpsCount];
	OpCounter sizesSmol[kKnownOpsCount];
	for (int i = 0; i < kKnownOpsCount; ++i)
	{
		counts[i].first = (SpvOp)i;
		counts[i].second = stats->opCounts[i];
		sizes[i].first = (SpvOp)i;
		sizes[i].second = stats->opSizes[i];
		sizesSmol[i].first = (SpvOp)i;
		sizesSmol[i].second = stats->smolOpSizes[i];
	}
	std::sort(counts, counts + kKnownOpsCount, CompareOpCounters);
	std::sort(sizes, sizes + kKnownOpsCount, CompareOpCounters);
	std::sort(sizesSmol, sizesSmol + kKnownOpsCount, CompareOpCounters);
	
	printf("Stats for %i SPIR-V inputs, total size %i words (%.1fKB):\n", (int)stats->inputCount, (int)stats->totalSize, stats->totalSize * 4.0f / 1024.0f);
	printf("Most occuring ops:\n");
	for (int i = 0; i < 30; ++i)
	{
		SpvOp op = counts[i].first;
		printf(" #%2i: %4i %-20s %4i (%4.1f%%)\n", i, op, kSpirvOpNames[op], (int)counts[i].second, (float)counts[i].second / (float)stats->totalOps * 100.0f);
	}
	printf("Largest total size of ops:\n");
	for (int i = 0; i < 30; ++i)
	{
		SpvOp op = sizes[i].first;
		printf(" #%2i: %-22s %6i (%4.1f%%) avg len %.1f\n",
			   i,
			   kSpirvOpNames[op],
			   (int)sizes[i].second*4,
			   (float)sizes[i].second / (float)stats->totalSize * 100.0f,
			   (float)sizes[i].second*4 / (float)stats->opCounts[op]
		);
	}
	printf("SMOL varint encoding counts per byte length:\n");
	printf("  B: %6s %6s %6s %6s\n", "Op", "Type", "Result", "Other");
	for (int i = 1; i < 6; ++i)
	{
		printf("  %i: %6i %6i %6i %6i\n", i, (int)stats->varintCountsOp[i], (int)stats->varintCountsType[i], (int)stats->varintCountsRes[i], (int)stats->varintCountsOther[i]);
	}
	printf("Largest total size of ops in SMOL:\n");
	for (int i = 0; i < 30; ++i)
	{
		SpvOp op = sizesSmol[i].first;
		printf(" #%2i: %-22s %6i (%4.1f%%) avg len %.1f\n",
			   i,
			   kSpirvOpNames[op],
			   (int)sizesSmol[i].second,
			   (float)sizesSmol[i].second / (float)stats->totalSizeSmol * 100.0f,
			   (float)sizesSmol[i].second / (float)stats->opCounts[op]
		);
	}	
}


// ------------------------------------------------------------------------------
// This software is available under 2 licenses -- choose whichever you prefer.
// ------------------------------------------------------------------------------
// ALTERNATIVE A - MIT License
// Copyright (c) 2016-2020 Aras Pranckevicius
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ------------------------------------------------------------------------------
// ALTERNATIVE B - Public Domain (www.unlicense.org)
// This is free and unencumbered software released into the public domain.
// Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
// software, either in source code form or as a compiled binary, for any purpose,
// commercial or non-commercial, and by any means.
// In jurisdictions that recognize copyright laws, the author or authors of this
// software dedicate any and all copyright interest in the software to the public
// domain. We make this dedication for the benefit of the public at large and to
// the detriment of our heirs and successors. We intend this dedication to be an
// overt act of relinquishment in perpetuity of all present and future rights to
// this software under copyright law.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ------------------------------------------------------------------------------
