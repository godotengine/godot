//  Copyright (c) Microsoft Corporation.  All rights reserved.

#ifndef DIRECTML_H
#define DIRECTML_H
#pragma once

#ifdef _GAMING_XBOX_SCARLETT
#include "d3d12_xs.h"
#elif _GAMING_XBOX_XBOXONE
#include "d3d12_x.h"
#else
#include "d3d12.h"
#endif

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES)

#ifndef DML_DECLARE_INTERFACE
#define DML_DECLARE_INTERFACE(iid) DECLSPEC_UUID(iid) DECLSPEC_NOVTABLE
#endif

#ifndef DML_TARGET_VERSION

#if !defined(NTDDI_VERSION) || defined(DML_TARGET_VERSION_USE_LATEST) // Use the latest if using redist or no Windows target set.
#define DML_TARGET_VERSION 0x6400
#elif defined(NTDDI_WIN10_ZN) && NTDDI_VERSION >= NTDDI_WIN10_ZN
#define DML_TARGET_VERSION 0x6000
#elif defined(NTDDI_WIN10_NI) && NTDDI_VERSION >= NTDDI_WIN10_NI
#define DML_TARGET_VERSION 0x5000
#elif defined(NTDDI_WIN10_CO) && NTDDI_VERSION >= NTDDI_WIN10_CO
#define DML_TARGET_VERSION 0x4000
#elif defined(NTDDI_WIN10_FE) && NTDDI_VERSION >= NTDDI_WIN10_FE
#define DML_TARGET_VERSION 0x3000
#elif defined(NTDDI_WIN10_VB) && NTDDI_VERSION >= NTDDI_WIN10_VB // Windows 10 2004 Update
#define DML_TARGET_VERSION 0x2000
#else // defined(NTDDI_WIN10_19H1) && NTDDI_VERSION >= NTDDI_WIN10_19H1 // Windows 10 1903 Update
#define DML_TARGET_VERSION 0x1000
#endif

#endif // !defined(DML_TARGET_VERSION)

// ===================================================================================================================
//   DirectML constants
// ===================================================================================================================

static const UINT DML_TENSOR_DIMENSION_COUNT_MAX = 5;
#if DML_TARGET_VERSION >= 0x3000
static const UINT DML_TENSOR_DIMENSION_COUNT_MAX1 = 8;
#endif

static const UINT DML_TEMPORARY_BUFFER_ALIGNMENT = 256;
static const UINT DML_PERSISTENT_BUFFER_ALIGNMENT = 256;

static const UINT DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT = 16;


// ===================================================================================================================
//   Interface declarations
// ===================================================================================================================

interface IDMLObject;
interface IDMLDevice;
interface IDMLDeviceChild;
interface IDMLPageable;
interface IDMLDispatchable;
interface IDMLOperator;
interface IDMLCompiledOperator;
interface IDMLOperatorInitializer;
interface IDMLBindingTable;
interface IDMLCommandRecorder;


// ===================================================================================================================
//   Tensor descriptions
// ===================================================================================================================

enum DML_TENSOR_DATA_TYPE
{
    DML_TENSOR_DATA_TYPE_UNKNOWN,
    DML_TENSOR_DATA_TYPE_FLOAT32,
    DML_TENSOR_DATA_TYPE_FLOAT16,
    DML_TENSOR_DATA_TYPE_UINT32,
    DML_TENSOR_DATA_TYPE_UINT16,
    DML_TENSOR_DATA_TYPE_UINT8,
    DML_TENSOR_DATA_TYPE_INT32,
    DML_TENSOR_DATA_TYPE_INT16,
    DML_TENSOR_DATA_TYPE_INT8,
    DML_TENSOR_DATA_TYPE_FLOAT64,
    DML_TENSOR_DATA_TYPE_UINT64,
    DML_TENSOR_DATA_TYPE_INT64,
#if DML_TARGET_VERSION >= 0x6300
    DML_TENSOR_DATA_TYPE_UINT4,
    DML_TENSOR_DATA_TYPE_INT4,
#endif // DML_TARGET_VERSION >= 0x6300
};

enum DML_TENSOR_TYPE
{
    DML_TENSOR_TYPE_INVALID,

    DML_TENSOR_TYPE_BUFFER,
};

enum DML_TENSOR_FLAGS
{
    DML_TENSOR_FLAG_NONE = 0x0,
    DML_TENSOR_FLAG_OWNED_BY_DML = 0x1,
};

DEFINE_ENUM_FLAG_OPERATORS(DML_TENSOR_FLAGS)

struct DML_BUFFER_TENSOR_DESC
{
    DML_TENSOR_DATA_TYPE DataType;
    DML_TENSOR_FLAGS Flags;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Sizes;
    _Field_size_opt_(DimensionCount) const UINT* Strides;
    UINT64 TotalTensorSizeInBytes;
    UINT GuaranteedBaseOffsetAlignment;
};

struct DML_TENSOR_DESC
{
    DML_TENSOR_TYPE Type;
    _Field_size_(_Inexpressible_("Dependent on tensor type")) const void* Desc;
};


// ===================================================================================================================
//   Operator types
// ===================================================================================================================

enum DML_OPERATOR_TYPE
{
    DML_OPERATOR_INVALID,

    DML_OPERATOR_ELEMENT_WISE_IDENTITY,
    DML_OPERATOR_ELEMENT_WISE_ABS,
    DML_OPERATOR_ELEMENT_WISE_ACOS,
    DML_OPERATOR_ELEMENT_WISE_ADD,
    DML_OPERATOR_ELEMENT_WISE_ASIN,
    DML_OPERATOR_ELEMENT_WISE_ATAN,
    DML_OPERATOR_ELEMENT_WISE_CEIL,
    DML_OPERATOR_ELEMENT_WISE_CLIP,
    DML_OPERATOR_ELEMENT_WISE_COS,
    DML_OPERATOR_ELEMENT_WISE_DIVIDE,
    DML_OPERATOR_ELEMENT_WISE_EXP,
    DML_OPERATOR_ELEMENT_WISE_FLOOR,
    DML_OPERATOR_ELEMENT_WISE_LOG,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR,
    DML_OPERATOR_ELEMENT_WISE_MAX,
    DML_OPERATOR_ELEMENT_WISE_MEAN,
    DML_OPERATOR_ELEMENT_WISE_MIN,
    DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
    DML_OPERATOR_ELEMENT_WISE_POW,
    DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW,
    DML_OPERATOR_ELEMENT_WISE_RECIP,
    DML_OPERATOR_ELEMENT_WISE_SIN,
    DML_OPERATOR_ELEMENT_WISE_SQRT,
    DML_OPERATOR_ELEMENT_WISE_SUBTRACT,
    DML_OPERATOR_ELEMENT_WISE_TAN,
    DML_OPERATOR_ELEMENT_WISE_THRESHOLD,
    DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR,
    DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR,
    DML_OPERATOR_ACTIVATION_ELU,
    DML_OPERATOR_ACTIVATION_HARDMAX,
    DML_OPERATOR_ACTIVATION_HARD_SIGMOID,
    DML_OPERATOR_ACTIVATION_IDENTITY,
    DML_OPERATOR_ACTIVATION_LEAKY_RELU,
    DML_OPERATOR_ACTIVATION_LINEAR,
    DML_OPERATOR_ACTIVATION_LOG_SOFTMAX,
    DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU,
    DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS,
    DML_OPERATOR_ACTIVATION_RELU,
    DML_OPERATOR_ACTIVATION_SCALED_ELU,
    DML_OPERATOR_ACTIVATION_SCALED_TANH,
    DML_OPERATOR_ACTIVATION_SIGMOID,
    DML_OPERATOR_ACTIVATION_SOFTMAX,
    DML_OPERATOR_ACTIVATION_SOFTPLUS,
    DML_OPERATOR_ACTIVATION_SOFTSIGN,
    DML_OPERATOR_ACTIVATION_TANH,
    DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU,
    DML_OPERATOR_CONVOLUTION,
    DML_OPERATOR_GEMM,
    DML_OPERATOR_REDUCE,
    DML_OPERATOR_AVERAGE_POOLING,
    DML_OPERATOR_LP_POOLING,
    DML_OPERATOR_MAX_POOLING,
    DML_OPERATOR_ROI_POOLING,
    DML_OPERATOR_SLICE,
    DML_OPERATOR_CAST,
    DML_OPERATOR_SPLIT,
    DML_OPERATOR_JOIN,
    DML_OPERATOR_PADDING,
    DML_OPERATOR_VALUE_SCALE_2D,
    DML_OPERATOR_UPSAMPLE_2D,
    DML_OPERATOR_GATHER,
    DML_OPERATOR_SPACE_TO_DEPTH,
    DML_OPERATOR_DEPTH_TO_SPACE,
    DML_OPERATOR_TILE,
    DML_OPERATOR_TOP_K,
    DML_OPERATOR_BATCH_NORMALIZATION,
    DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION,
    DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION,
    DML_OPERATOR_LP_NORMALIZATION,
    DML_OPERATOR_RNN,
    DML_OPERATOR_LSTM,
    DML_OPERATOR_GRU,

#if DML_TARGET_VERSION >= 0x2000
    DML_OPERATOR_ELEMENT_WISE_SIGN,
    DML_OPERATOR_ELEMENT_WISE_IS_NAN,
    DML_OPERATOR_ELEMENT_WISE_ERF,
    DML_OPERATOR_ELEMENT_WISE_SINH,
    DML_OPERATOR_ELEMENT_WISE_COSH,
    DML_OPERATOR_ELEMENT_WISE_TANH,
    DML_OPERATOR_ELEMENT_WISE_ASINH,
    DML_OPERATOR_ELEMENT_WISE_ACOSH,
    DML_OPERATOR_ELEMENT_WISE_ATANH,
    DML_OPERATOR_ELEMENT_WISE_IF,
    DML_OPERATOR_ELEMENT_WISE_ADD1,
    DML_OPERATOR_ACTIVATION_SHRINK,
    DML_OPERATOR_MAX_POOLING1,
    DML_OPERATOR_MAX_UNPOOLING,
    DML_OPERATOR_DIAGONAL_MATRIX,
    DML_OPERATOR_SCATTER_ELEMENTS,
    DML_OPERATOR_SCATTER = DML_OPERATOR_SCATTER_ELEMENTS, // Alias name for backwards compatibility.
    DML_OPERATOR_ONE_HOT,
    DML_OPERATOR_RESAMPLE,
#endif // DML_TARGET_VERSION >= 0x2000

#if DML_TARGET_VERSION >= 0x2100
    DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT,
    DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT,
    DML_OPERATOR_ELEMENT_WISE_ROUND,
    DML_OPERATOR_ELEMENT_WISE_IS_INFINITY,
    DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE,
    DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR,
    DML_OPERATOR_FILL_VALUE_CONSTANT,
    DML_OPERATOR_FILL_VALUE_SEQUENCE,
    DML_OPERATOR_CUMULATIVE_SUMMATION,
    DML_OPERATOR_REVERSE_SUBSEQUENCES,
    DML_OPERATOR_GATHER_ELEMENTS,
    DML_OPERATOR_GATHER_ND,
    DML_OPERATOR_SCATTER_ND,
    DML_OPERATOR_MAX_POOLING2,
    DML_OPERATOR_SLICE1,
    DML_OPERATOR_TOP_K1,
    DML_OPERATOR_DEPTH_TO_SPACE1,
    DML_OPERATOR_SPACE_TO_DEPTH1,
    DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1,
    DML_OPERATOR_RESAMPLE1,
    DML_OPERATOR_MATRIX_MULTIPLY_INTEGER,
    DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY,
    DML_OPERATOR_CONVOLUTION_INTEGER,
    DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION,
#endif // DML_TARGET_VERSION >= 0x2100

#if DML_TARGET_VERSION >= 0x3000
    DML_OPERATOR_ELEMENT_WISE_BIT_AND,
    DML_OPERATOR_ELEMENT_WISE_BIT_OR,
    DML_OPERATOR_ELEMENT_WISE_BIT_XOR,
    DML_OPERATOR_ELEMENT_WISE_BIT_NOT,
    DML_OPERATOR_ELEMENT_WISE_BIT_COUNT,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL,
    DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL,
    DML_OPERATOR_ACTIVATION_CELU,
    DML_OPERATOR_ACTIVATION_RELU_GRAD,
    DML_OPERATOR_AVERAGE_POOLING_GRAD,
    DML_OPERATOR_MAX_POOLING_GRAD,
    DML_OPERATOR_RANDOM_GENERATOR,
    DML_OPERATOR_NONZERO_COORDINATES,
    DML_OPERATOR_RESAMPLE_GRAD,
    DML_OPERATOR_SLICE_GRAD,
    DML_OPERATOR_ADAM_OPTIMIZER,
    DML_OPERATOR_ARGMIN,
    DML_OPERATOR_ARGMAX,
    DML_OPERATOR_ROI_ALIGN,
    DML_OPERATOR_GATHER_ND1,
#endif // DML_TARGET_VERSION >= 0x3000

#if DML_TARGET_VERSION >= 0x3100
    DML_OPERATOR_ELEMENT_WISE_ATAN_YX,
    DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD,
    DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE,
    DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD,
    DML_OPERATOR_CUMULATIVE_PRODUCT,
    DML_OPERATOR_BATCH_NORMALIZATION_GRAD,
#endif // DML_TARGET_VERSION >= 0x3100

#if DML_TARGET_VERSION >= 0x4000
    DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD,
    DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR,
    DML_OPERATOR_ROI_ALIGN1,
#endif // DML_TARGET_VERSION >= 0x4000

#if DML_TARGET_VERSION >= 0x4100
    DML_OPERATOR_ROI_ALIGN_GRAD,
    DML_OPERATOR_BATCH_NORMALIZATION_TRAINING,
    DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD,
#endif // DML_TARGET_VERSION >= 0x4100

#if DML_TARGET_VERSION >= 0x5000
    DML_OPERATOR_ELEMENT_WISE_CLIP1,
    DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1,
    DML_OPERATOR_PADDING1,
    DML_OPERATOR_ELEMENT_WISE_NEGATE,
#endif // DML_TARGET_VERSION >= 0x5000

#if DML_TARGET_VERSION >= 0x5100
    DML_OPERATOR_ACTIVATION_GELU,
    DML_OPERATOR_ACTIVATION_SOFTMAX1,
    DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1,
    DML_OPERATOR_ACTIVATION_HARDMAX1,
    DML_OPERATOR_RESAMPLE2,
    DML_OPERATOR_RESAMPLE_GRAD1,
    DML_OPERATOR_DIAGONAL_MATRIX1,
#endif // DML_TARGET_VERSION >= 0x5100

#if DML_TARGET_VERSION >= 0x6100
    DML_OPERATOR_MULTIHEAD_ATTENTION,
#endif // DML_TARGET_VERSION >= 0x6100

#if DML_TARGET_VERSION >= 0x6200
    DML_OPERATOR_LP_POOLING1,
    DML_OPERATOR_AVERAGE_POOLING1,
    DML_OPERATOR_ACTIVATION_SWISH,
    DML_OPERATOR_ACTIVATION_HARD_SWISH,
    DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING,
    DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT,
#endif // DML_TARGET_VERSION >= 0x6200

#if DML_TARGET_VERSION >= 0x6300
    DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2,
    DML_OPERATOR_MULTIHEAD_ATTENTION1,
    DML_OPERATOR_QUANTIZE,
    DML_OPERATOR_DEQUANTIZE,
#endif // DML_TARGET_VERSION >= 0x6300

#if DML_TARGET_VERSION >= 0x6400
    DML_OPERATOR_RESAMPLE3,
    DML_OPERATOR_FOLD,
    DML_OPERATOR_UNFOLD,
#endif // DML_TARGET_VERSION >= 0x6400
};

// ===================================================================================================================
//   Operator enumerations and structures
// ===================================================================================================================

enum DML_REDUCE_FUNCTION
{
    DML_REDUCE_FUNCTION_ARGMAX,
    DML_REDUCE_FUNCTION_ARGMIN,
    DML_REDUCE_FUNCTION_AVERAGE,
    DML_REDUCE_FUNCTION_L1,
    DML_REDUCE_FUNCTION_L2,
    DML_REDUCE_FUNCTION_LOG_SUM,
    DML_REDUCE_FUNCTION_LOG_SUM_EXP,
    DML_REDUCE_FUNCTION_MAX,
    DML_REDUCE_FUNCTION_MIN,
    DML_REDUCE_FUNCTION_MULTIPLY,
    DML_REDUCE_FUNCTION_SUM,
    DML_REDUCE_FUNCTION_SUM_SQUARE,
};

enum DML_MATRIX_TRANSFORM
{
    DML_MATRIX_TRANSFORM_NONE,
    DML_MATRIX_TRANSFORM_TRANSPOSE,
};

enum DML_CONVOLUTION_MODE
{
    DML_CONVOLUTION_MODE_CONVOLUTION,
    DML_CONVOLUTION_MODE_CROSS_CORRELATION,
};

enum DML_CONVOLUTION_DIRECTION
{
    DML_CONVOLUTION_DIRECTION_FORWARD,
    DML_CONVOLUTION_DIRECTION_BACKWARD,
};

enum DML_PADDING_MODE
{
    DML_PADDING_MODE_CONSTANT,
    DML_PADDING_MODE_EDGE,
    DML_PADDING_MODE_REFLECTION,

#if DML_TARGET_VERSION >= 0x3000
    DML_PADDING_MODE_SYMMETRIC,
#endif

#if DML_TARGET_VERSION >= 0x6400
    DML_PADDING_MODE_WRAP,
#endif
};

enum DML_INTERPOLATION_MODE
{
    DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR,
    DML_INTERPOLATION_MODE_LINEAR,
};

struct DML_SCALE_BIAS
{
    FLOAT Scale;
    FLOAT Bias;
};

struct DML_SIZE_2D
{
    UINT Width;
    UINT Height;
};

enum DML_RECURRENT_NETWORK_DIRECTION
{
    DML_RECURRENT_NETWORK_DIRECTION_FORWARD,
    DML_RECURRENT_NETWORK_DIRECTION_BACKWARD,
    DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL,
};

#if DML_TARGET_VERSION >= 0x2100

enum DML_ROUNDING_MODE
{
    DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN,
    DML_ROUNDING_MODE_TOWARD_ZERO,
    DML_ROUNDING_MODE_TOWARD_INFINITY,
};

enum DML_IS_INFINITY_MODE
{
    DML_IS_INFINITY_MODE_EITHER = 0,
    DML_IS_INFINITY_MODE_POSITIVE = 1,
    DML_IS_INFINITY_MODE_NEGATIVE = 2,
};

enum DML_AXIS_DIRECTION
{
    DML_AXIS_DIRECTION_INCREASING = 0,
    DML_AXIS_DIRECTION_DECREASING = 1,
};

enum DML_DEPTH_SPACE_ORDER
{
    DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW,
    DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH,
};

union DML_SCALAR_UNION
{
    BYTE   Bytes[8];
    INT8   Int8;
    UINT8  UInt8;
    INT16  Int16;
    UINT16 UInt16;
    INT32  Int32;
    UINT32 UInt32;
    INT64  Int64;
    UINT64 UInt64;
    FLOAT  Float32;
    DOUBLE Float64;
};

#endif // DML_TARGET_VERSION >= 0x2100

#if DML_TARGET_VERSION >= 0x3000

enum DML_RANDOM_GENERATOR_TYPE
{
    DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10
};

#endif // DML_TARGET_VERSION >= 0x3000

#if DML_TARGET_VERSION >= 0x6100

enum DML_MULTIHEAD_ATTENTION_MASK_TYPE
{
    DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE,
    DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH,
    DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_END_START,
    DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_QUERY_SEQUENCE_LENGTH_START_END,
    DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN,
};

#endif // DML_TARGET_VERSION >= 0x6100

#if DML_TARGET_VERSION >= 0x6300

enum DML_QUANTIZATION_TYPE
{
    DML_QUANTIZATION_TYPE_NONE,
    DML_QUANTIZATION_TYPE_SCALE,
    DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT,
};

#endif // DML_TARGET_VERSION >= 0x6300

// ===================================================================================================================
//   Operator descriptions
// ===================================================================================================================

struct DML_OPERATOR_DESC
{
    DML_OPERATOR_TYPE Type;
    _Field_size_(_Inexpressible_("Dependent on operator type")) const void* Desc;
};

struct DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_ABS_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_ACOS_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_ADD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_ADD1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_OPERATOR_DESC* FusedActivation;
};

struct DML_ELEMENT_WISE_ASIN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_ATAN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_CEIL_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_CLIP_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
    FLOAT Min;
    FLOAT Max;
};

struct DML_ELEMENT_WISE_COS_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_EXP_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_LOG_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_MAX_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_MEAN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_MIN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_POW_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* ExponentTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
    FLOAT Exponent;
};

struct DML_ELEMENT_WISE_RECIP_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_SIN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_SQRT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_TAN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_THRESHOLD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
    FLOAT Min;
};

struct DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* ScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* ZeroPointTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* ScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* ZeroPointTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_ELU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
};

struct DML_ACTIVATION_HARDMAX_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
    FLOAT Beta;
};

struct DML_ACTIVATION_IDENTITY_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
};

struct DML_ACTIVATION_LINEAR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
    FLOAT Beta;
};

struct DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* SlopeTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
    FLOAT Beta;
};

struct DML_ACTIVATION_RELU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
    FLOAT Gamma;
};

struct DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
    FLOAT Beta;
};

struct DML_ACTIVATION_SIGMOID_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_SOFTMAX_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Steepness;
};

struct DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_TANH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
};

struct DML_CONVOLUTION_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* FilterTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_CONVOLUTION_MODE Mode;
    DML_CONVOLUTION_DIRECTION Direction;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* Dilations;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    _Field_size_(DimensionCount) const UINT* OutputPadding;
    UINT GroupCount;
    _Maybenull_ const DML_OPERATOR_DESC* FusedActivation;
};

struct DML_GEMM_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    _Maybenull_ const DML_TENSOR_DESC* CTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_MATRIX_TRANSFORM TransA;
    DML_MATRIX_TRANSFORM TransB;
    FLOAT Alpha;
    FLOAT Beta;
    _Maybenull_ const DML_OPERATOR_DESC* FusedActivation;
};

struct DML_REDUCE_OPERATOR_DESC
{
    DML_REDUCE_FUNCTION Function;
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT AxisCount;
    _Field_size_(AxisCount) const UINT* Axes;
};

struct DML_AVERAGE_POOLING_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    BOOL IncludePadding;
};

struct DML_LP_POOLING_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    UINT P;
};

struct DML_MAX_POOLING_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
};

struct DML_ROI_POOLING_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* ROITensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT SpatialScale;
    DML_SIZE_2D PooledSize;
};

struct DML_SLICE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Offsets;
    _Field_size_(DimensionCount) const UINT* Sizes;
    _Field_size_(DimensionCount) const UINT* Strides;
};

struct DML_CAST_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_SPLIT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    UINT OutputCount;
    _Field_size_(OutputCount) const DML_TENSOR_DESC* OutputTensors;
    UINT Axis;
};

struct DML_JOIN_OPERATOR_DESC
{
    UINT InputCount;
    _Field_size_(InputCount) const DML_TENSOR_DESC* InputTensors;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
};

struct DML_PADDING_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_PADDING_MODE PaddingMode;
    FLOAT PaddingValue;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
};

struct DML_VALUE_SCALE_2D_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Scale;
    UINT ChannelCount;
    _Field_size_(ChannelCount) const FLOAT* Bias;
};

struct DML_UPSAMPLE_2D_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_SIZE_2D ScaleSize;
    DML_INTERPOLATION_MODE InterpolationMode;
};

struct DML_GATHER_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* IndicesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
    UINT IndexDimensions;
};

struct DML_SPACE_TO_DEPTH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT BlockSize;
};

struct DML_DEPTH_TO_SPACE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT BlockSize;
};

struct DML_TILE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT RepeatsCount;
    _Field_size_(RepeatsCount) const UINT* Repeats;
};

struct DML_TOP_K_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputValueTensor;
    const DML_TENSOR_DESC* OutputIndexTensor;
    UINT Axis;
    UINT K;
};

struct DML_BATCH_NORMALIZATION_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* MeanTensor;
    const DML_TENSOR_DESC* VarianceTensor;
    const DML_TENSOR_DESC* ScaleTensor;
    const DML_TENSOR_DESC* BiasTensor;
    const DML_TENSOR_DESC* OutputTensor;
    BOOL Spatial;
    FLOAT Epsilon;
    _Maybenull_ const DML_OPERATOR_DESC* FusedActivation;
};

struct DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    _Maybenull_ const DML_TENSOR_DESC* ScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    const DML_TENSOR_DESC* OutputTensor;
    BOOL CrossChannel;
    BOOL NormalizeVariance;
    FLOAT Epsilon;
    _Maybenull_ const DML_OPERATOR_DESC* FusedActivation;
};

struct DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    BOOL CrossChannel;
    UINT LocalSize;
    FLOAT Alpha;
    FLOAT Beta;
    FLOAT Bias;
};

struct DML_LP_NORMALIZATION_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
    FLOAT Epsilon;
    UINT P;
};

struct DML_RNN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* WeightTensor;
    const DML_TENSOR_DESC* RecurrenceTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* HiddenInitTensor;
    _Maybenull_ const DML_TENSOR_DESC* SequenceLengthsTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputSequenceTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputSingleTensor;
    UINT ActivationDescCount;
    _Field_size_(ActivationDescCount) const DML_OPERATOR_DESC* ActivationDescs;
    DML_RECURRENT_NETWORK_DIRECTION Direction;
};

struct DML_LSTM_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* WeightTensor;
    const DML_TENSOR_DESC* RecurrenceTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* HiddenInitTensor;
    _Maybenull_ const DML_TENSOR_DESC* CellMemInitTensor;
    _Maybenull_ const DML_TENSOR_DESC* SequenceLengthsTensor;
    _Maybenull_ const DML_TENSOR_DESC* PeepholeTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputSequenceTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputSingleTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputCellSingleTensor;
    UINT ActivationDescCount;
    _Field_size_(ActivationDescCount) const DML_OPERATOR_DESC* ActivationDescs;
    DML_RECURRENT_NETWORK_DIRECTION Direction;
    float ClipThreshold;
    BOOL UseClipThreshold;
    BOOL CoupleInputForget;
};

struct DML_GRU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* WeightTensor;
    const DML_TENSOR_DESC* RecurrenceTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* HiddenInitTensor;
    _Maybenull_ const DML_TENSOR_DESC* SequenceLengthsTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputSequenceTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputSingleTensor;
    UINT ActivationDescCount;
    _Field_size_(ActivationDescCount) const DML_OPERATOR_DESC* ActivationDescs;
    DML_RECURRENT_NETWORK_DIRECTION Direction;
    BOOL LinearBeforeReset;
};

#if DML_TARGET_VERSION >= 0x2000

struct DML_ELEMENT_WISE_SIGN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_ERF_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_SINH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_COSH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_TANH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_ASINH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_ATANH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
};

struct DML_ELEMENT_WISE_IF_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ConditionTensor;
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_SHRINK_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Bias;
    FLOAT Threshold;
};

struct DML_MAX_POOLING1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputIndicesTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
};

struct DML_MAX_UNPOOLING_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* IndicesTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_DIAGONAL_MATRIX_OPERATOR_DESC
{
    const DML_TENSOR_DESC* OutputTensor;
    INT Offset;
    FLOAT Value;
};

struct DML_SCATTER_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* IndicesTensor;
    const DML_TENSOR_DESC* UpdatesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
};

struct DML_ONE_HOT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* IndicesTensor;
    const DML_TENSOR_DESC* ValuesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
};

struct DML_RESAMPLE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_INTERPOLATION_MODE InterpolationMode;
    UINT ScaleCount;
    _Field_size_(ScaleCount) const FLOAT* Scales;
};

#endif // DML_TARGET_VERSION >= 0x2000

#if DML_TARGET_VERSION >= 0x2100

struct DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_ROUND_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_ROUNDING_MODE RoundingMode;
};

struct DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_IS_INFINITY_MODE InfinityMode;
};

struct DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_FILL_VALUE_CONSTANT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* OutputTensor;
    DML_TENSOR_DATA_TYPE ValueDataType;
    DML_SCALAR_UNION Value;
};

struct DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* OutputTensor;
    DML_TENSOR_DATA_TYPE ValueDataType;
    DML_SCALAR_UNION ValueStart;
    DML_SCALAR_UNION ValueDelta;
};

struct DML_CUMULATIVE_SUMMATION_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
    DML_AXIS_DIRECTION AxisDirection;
    BOOL HasExclusiveSum;
};

struct DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* SequenceLengthsTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
};

struct DML_GATHER_ELEMENTS_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* IndicesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
};

// Alias existing operator, symmetric with DML_GATHER_ELEMENTS_OPERATOR_DESC.
using DML_SCATTER_ELEMENTS_OPERATOR_DESC = DML_SCATTER_OPERATOR_DESC;

struct DML_GATHER_ND_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* IndicesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT InputDimensionCount;
    UINT IndicesDimensionCount;
};

struct DML_SCATTER_ND_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* IndicesTensor;
    const DML_TENSOR_DESC* UpdatesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT InputDimensionCount;
    UINT IndicesDimensionCount;
};

struct DML_MAX_POOLING2_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputIndicesTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    _Field_size_(DimensionCount) const UINT* Dilations;
};

struct DML_SLICE1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* InputWindowOffsets;
    _Field_size_(DimensionCount) const UINT* InputWindowSizes;
    _Field_size_(DimensionCount) const INT* InputWindowStrides;
};

struct DML_TOP_K1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputValueTensor;
    const DML_TENSOR_DESC* OutputIndexTensor;
    UINT Axis;
    UINT K;
    DML_AXIS_DIRECTION AxisDirection;
};

struct DML_DEPTH_TO_SPACE1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT BlockSize;
    DML_DEPTH_SPACE_ORDER Order;
};

struct DML_SPACE_TO_DEPTH1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT BlockSize;
    DML_DEPTH_SPACE_ORDER Order;
};

struct DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    _Maybenull_ const DML_TENSOR_DESC* ScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT AxisCount;
    _Field_size_(AxisCount) const UINT* Axes;
    BOOL NormalizeVariance;
    FLOAT Epsilon;
    _Maybenull_ const DML_OPERATOR_DESC* FusedActivation;
};

struct DML_RESAMPLE1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_INTERPOLATION_MODE InterpolationMode;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const FLOAT* Scales;
    _Field_size_(DimensionCount) const FLOAT* InputPixelOffsets;
    _Field_size_(DimensionCount) const FLOAT* OutputPixelOffsets;
};

struct DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    _Maybenull_ const DML_TENSOR_DESC* AZeroPointTensor;
    const DML_TENSOR_DESC* BTensor;
    _Maybenull_ const DML_TENSOR_DESC* BZeroPointTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* AScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* AZeroPointTensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* BScaleTensor; 
    _Maybenull_ const DML_TENSOR_DESC* BZeroPointTensor;
    const DML_TENSOR_DESC* OutputScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputZeroPointTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_CONVOLUTION_INTEGER_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputZeroPointTensor;
    const DML_TENSOR_DESC* FilterTensor;
    _Maybenull_ const DML_TENSOR_DESC* FilterZeroPointTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* Dilations;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    UINT GroupCount;
};

struct DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputZeroPointTensor;
    const DML_TENSOR_DESC* FilterTensor;
    const DML_TENSOR_DESC* FilterScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* FilterZeroPointTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    const DML_TENSOR_DESC* OutputScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputZeroPointTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* Dilations;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    UINT GroupCount;
};

#endif // DML_TARGET_VERSION >= 0x2100

#if DML_TARGET_VERSION >= 0x3000

struct DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_CELU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
};

struct DML_ACTIVATION_RELU_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
};

struct DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    BOOL IncludePadding;
};

struct DML_MAX_POOLING_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    _Field_size_(DimensionCount) const UINT* Dilations;
};

struct DML_RANDOM_GENERATOR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputStateTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputStateTensor;
    DML_RANDOM_GENERATOR_TYPE Type;
};

struct DML_NONZERO_COORDINATES_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputCountTensor;
    const DML_TENSOR_DESC* OutputCoordinatesTensor;
};

struct DML_RESAMPLE_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    DML_INTERPOLATION_MODE InterpolationMode;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const FLOAT* Scales;
    _Field_size_(DimensionCount) const FLOAT* InputPixelOffsets;
    _Field_size_(DimensionCount) const FLOAT* OutputPixelOffsets;
};

struct DML_SLICE_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* InputWindowOffsets;
    _Field_size_(DimensionCount) const UINT* InputWindowSizes;
    _Field_size_(DimensionCount) const INT* InputWindowStrides;
};

struct DML_ADAM_OPTIMIZER_OPERATOR_DESC
{ 
    const DML_TENSOR_DESC* InputParametersTensor;
    const DML_TENSOR_DESC* InputFirstMomentTensor;
    const DML_TENSOR_DESC* InputSecondMomentTensor;
    const DML_TENSOR_DESC* GradientTensor;
    const DML_TENSOR_DESC* TrainingStepTensor;
    const DML_TENSOR_DESC* OutputParametersTensor;
    const DML_TENSOR_DESC* OutputFirstMomentTensor;
    const DML_TENSOR_DESC* OutputSecondMomentTensor;
    FLOAT LearningRate;
    FLOAT Beta1;
    FLOAT Beta2;
    FLOAT Epsilon;
};

struct DML_ARGMIN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT AxisCount;
    _Field_size_(AxisCount) const UINT* Axes;
    DML_AXIS_DIRECTION AxisDirection;
};

struct DML_ARGMAX_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT AxisCount;
    _Field_size_(AxisCount) const UINT* Axes;
    DML_AXIS_DIRECTION AxisDirection;
};

struct DML_ROI_ALIGN_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* ROITensor;
    const DML_TENSOR_DESC* BatchIndicesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_REDUCE_FUNCTION ReductionFunction;
    DML_INTERPOLATION_MODE InterpolationMode;
    FLOAT SpatialScaleX;
    FLOAT SpatialScaleY;
    FLOAT OutOfBoundsInputValue;
    UINT MinimumSamplesPerOutput;
    UINT MaximumSamplesPerOutput;
};

struct DML_GATHER_ND1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* IndicesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT InputDimensionCount;
    UINT IndicesDimensionCount;
    UINT BatchDimensionCount;
};

#endif // DML_TARGET_VERSION >= 0x3000

#if DML_TARGET_VERSION >= 0x3100

struct DML_ELEMENT_WISE_ATAN_YX_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    FLOAT Min;
    FLOAT Max;
};

struct DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    BOOL CrossChannel;
    UINT LocalSize;
    FLOAT Alpha;
    FLOAT Beta;
    FLOAT Bias;
};

struct DML_CUMULATIVE_PRODUCT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT Axis;
    DML_AXIS_DIRECTION AxisDirection;
    BOOL HasExclusiveProduct;
};

struct DML_BATCH_NORMALIZATION_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* MeanTensor;
    const DML_TENSOR_DESC* VarianceTensor;
    const DML_TENSOR_DESC* ScaleTensor;

    const DML_TENSOR_DESC* OutputGradientTensor;
    const DML_TENSOR_DESC* OutputScaleGradientTensor;
    const DML_TENSOR_DESC* OutputBiasGradientTensor;

    FLOAT Epsilon;
};

#endif // DML_TARGET_VERSION >= 0x3100

#if DML_TARGET_VERSION >= 0x4000
struct DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* AScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* AZeroPointTensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* BScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* BZeroPointTensor;
    const DML_TENSOR_DESC* OutputScaleTensor;                   // This is an input tensor
    _Maybenull_ const DML_TENSOR_DESC* OutputZeroPointTensor;   // This is an input tensor
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    const DML_TENSOR_DESC* OutputScaleTensor;                   // This is an output tensor
    const DML_TENSOR_DESC* OutputZeroPointTensor;               // This is an output tensor
};

struct DML_ROI_ALIGN1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* ROITensor;
    const DML_TENSOR_DESC* BatchIndicesTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_REDUCE_FUNCTION ReductionFunction;
    DML_INTERPOLATION_MODE InterpolationMode;
    FLOAT SpatialScaleX;
    FLOAT SpatialScaleY;
    FLOAT InputPixelOffset;
    FLOAT OutputPixelOffset;
    FLOAT OutOfBoundsInputValue;
    UINT MinimumSamplesPerOutput;
    UINT MaximumSamplesPerOutput;
    BOOL AlignRegionsToCorners;
};

#endif // DML_TARGET_VERSION >= 0x4000

#if DML_TARGET_VERSION >= 0x4100

struct DML_ROI_ALIGN_GRAD_OPERATOR_DESC
{
    _Maybenull_ const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* ROITensor;
    const DML_TENSOR_DESC* BatchIndicesTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputGradientTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputROIGradientTensor;
    DML_REDUCE_FUNCTION ReductionFunction;
    DML_INTERPOLATION_MODE InterpolationMode;
    FLOAT SpatialScaleX;
    FLOAT SpatialScaleY;
    FLOAT InputPixelOffset;
    FLOAT OutputPixelOffset;
    UINT MinimumSamplesPerOutput;
    UINT MaximumSamplesPerOutput;
    BOOL AlignRegionsToCorners;
};

struct DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* ScaleTensor;
    const DML_TENSOR_DESC* BiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* FusedAddTensor;
    const DML_TENSOR_DESC* OutputTensor;
    const DML_TENSOR_DESC* OutputMeanTensor;
    const DML_TENSOR_DESC* OutputVarianceTensor;
    FLOAT Epsilon;
    _Maybenull_ const DML_OPERATOR_DESC* FusedActivation;
};

struct DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* MeanTensor;
    const DML_TENSOR_DESC* VarianceTensor;
    const DML_TENSOR_DESC* ScaleTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    const DML_TENSOR_DESC* OutputScaleGradientTensor;
    const DML_TENSOR_DESC* OutputBiasGradientTensor;
    FLOAT Epsilon;
};

#endif // DML_TARGET_VERSION >= 0x4100

#if DML_TARGET_VERSION >= 0x5000

struct DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_SCALE_BIAS* ScaleBias;
    DML_TENSOR_DATA_TYPE MinMaxDataType;
    DML_SCALAR_UNION Min;
    DML_SCALAR_UNION Max;
};

struct DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    DML_TENSOR_DATA_TYPE MinMaxDataType;
    DML_SCALAR_UNION Min;
    DML_SCALAR_UNION Max;
};

struct DML_PADDING1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_PADDING_MODE PaddingMode;
    DML_TENSOR_DATA_TYPE PaddingValueDataType;
    DML_SCALAR_UNION PaddingValue;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
};

struct DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

#endif // DML_TARGET_VERSION >= 0x5000

#if DML_TARGET_VERSION >= 0x5100

struct DML_ACTIVATION_GELU_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT AxisCount;
    _Field_size_(AxisCount) const UINT* Axes;
};

struct DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT AxisCount;
    _Field_size_(AxisCount) const UINT* Axes;
};

struct DML_ACTIVATION_HARDMAX1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT AxisCount;
    _Field_size_(AxisCount) const UINT* Axes;
};

struct DML_RESAMPLE2_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_INTERPOLATION_MODE InterpolationMode;
    DML_AXIS_DIRECTION RoundingDirection;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const FLOAT* Scales;
    _Field_size_(DimensionCount) const FLOAT* InputPixelOffsets;
    _Field_size_(DimensionCount) const FLOAT* OutputPixelOffsets;
};

struct DML_RESAMPLE_GRAD1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputGradientTensor;
    const DML_TENSOR_DESC* OutputGradientTensor;
    DML_INTERPOLATION_MODE InterpolationMode;
    DML_AXIS_DIRECTION RoundingDirection;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const FLOAT* Scales;
    _Field_size_(DimensionCount) const FLOAT* InputPixelOffsets;
    _Field_size_(DimensionCount) const FLOAT* OutputPixelOffsets;
};

struct DML_DIAGONAL_MATRIX1_OPERATOR_DESC
{
    _Maybenull_ const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_TENSOR_DATA_TYPE ValueDataType;
    DML_SCALAR_UNION Value;
    INT DiagonalFillBegin;
    INT DiagonalFillEnd;
};

#endif // DML_TARGET_VERSION >= 0x5100

#if DML_TARGET_VERSION >= 0x6100

struct DML_MULTIHEAD_ATTENTION_OPERATOR_DESC
{
    _Maybenull_ const DML_TENSOR_DESC* QueryTensor;
    _Maybenull_ const DML_TENSOR_DESC* KeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* ValueTensor;
    _Maybenull_ const DML_TENSOR_DESC* StackedQueryKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* StackedKeyValueTensor;
    _Maybenull_ const DML_TENSOR_DESC* StackedQueryKeyValueTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* MaskTensor;
    _Maybenull_ const DML_TENSOR_DESC* RelativePositionBiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* PastKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* PastValueTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputPresentKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputPresentValueTensor;
    FLOAT Scale;
    FLOAT MaskFilterValue;
    UINT HeadCount;
    DML_MULTIHEAD_ATTENTION_MASK_TYPE MaskType;
};

#endif // DML_TARGET_VERSION >= 0x6100

#if DML_TARGET_VERSION >= 0x6200

struct DML_LP_POOLING1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    _Field_size_(DimensionCount) const UINT* Dilations;
    UINT P;
};

struct DML_AVERAGE_POOLING1_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    _Field_size_(DimensionCount) const UINT* Dilations;
    BOOL IncludePadding;
};

struct DML_ACTIVATION_SWISH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT SigmoidInputScale;
};

struct DML_ACTIVATION_HARD_SWISH_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT Alpha;
    FLOAT Beta;
};

struct DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* InputScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputZeroPointTensor;
    const DML_TENSOR_DESC* OutputScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputZeroPointTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* Strides;
    _Field_size_(DimensionCount) const UINT* WindowSize;
    _Field_size_(DimensionCount) const UINT* StartPadding;
    _Field_size_(DimensionCount) const UINT* EndPadding;
    _Field_size_(DimensionCount) const UINT* Dilations;
    BOOL IncludePadding;
};

struct DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC
{
    const DML_TENSOR_DESC* ATensor;
    const DML_TENSOR_DESC* AScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* AZeroPointTensor;
    const DML_TENSOR_DESC* BTensor;
    const DML_TENSOR_DESC* BScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* BZeroPointTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    const DML_TENSOR_DESC* OutputTensor;
};

#endif // DML_TARGET_VERSION >= 0x6200

#if DML_TARGET_VERSION >= 0x6300

struct DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    _Maybenull_ const DML_TENSOR_DESC* ScaleTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT AxisCount;
    _Field_size_(AxisCount) const UINT* Axes;
    BOOL UseMean;
    BOOL UseVariance;
    FLOAT Epsilon;
    _Maybenull_ const DML_OPERATOR_DESC* FusedActivation;
};

struct DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC
{
    _Maybenull_ const DML_TENSOR_DESC* QueryTensor;
    _Maybenull_ const DML_TENSOR_DESC* KeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* ValueTensor;
    _Maybenull_ const DML_TENSOR_DESC* StackedQueryKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* StackedKeyValueTensor;
    _Maybenull_ const DML_TENSOR_DESC* StackedQueryKeyValueTensor;
    _Maybenull_ const DML_TENSOR_DESC* BiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* MaskTensor;
    _Maybenull_ const DML_TENSOR_DESC* RelativePositionBiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* PastKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* PastValueTensor;
    _Maybenull_ const DML_TENSOR_DESC* PastSequenceLengthsTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputPresentKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputPresentValueTensor;
    FLOAT Scale;
    FLOAT MaskFilterValue;
    UINT QueryHeadCount;
    UINT KeyValueHeadCount;
    DML_MULTIHEAD_ATTENTION_MASK_TYPE MaskType;
};

struct DML_QUANTIZE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    DML_QUANTIZATION_TYPE QuantizationType;
    UINT QuantizationTensorCount;
    _Field_size_(QuantizationTensorCount) const DML_TENSOR_DESC* QuantizationTensors;
    const DML_TENSOR_DESC* OutputTensor;
};

struct DML_DEQUANTIZE_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    DML_QUANTIZATION_TYPE QuantizationType;
    UINT QuantizationTensorCount;
    _Field_size_(QuantizationTensorCount) const DML_TENSOR_DESC* QuantizationTensors;
    const DML_TENSOR_DESC* OutputTensor;
};

#endif // DML_TARGET_VERSION >= 0x6300

#if DML_TARGET_VERSION >= 0x6400

struct DML_RESAMPLE3_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    DML_INTERPOLATION_MODE InterpolationMode;
    DML_AXIS_DIRECTION RoundingDirection;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const FLOAT* Scales;
    _Field_size_(DimensionCount) const FLOAT* InputPixelOffsets;
    _Field_size_(DimensionCount) const FLOAT* OutputPixelOffsets;
    BOOL Antialiased;
};

struct DML_FOLD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* WindowSizes; // Size of the extracted patch
    _Field_size_(DimensionCount) const UINT* Strides; // Step size of the extracted patches
    _Field_size_(DimensionCount) const UINT* Dilations; // Dialations of the extracted patch
    _Field_size_(DimensionCount) const UINT* StartPadding; // Start padding of the "source tensor"
    _Field_size_(DimensionCount) const UINT* EndPadding; // End padding of the "source tensor"
};

struct DML_UNFOLD_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputTensor;
    const DML_TENSOR_DESC* OutputTensor;
    UINT DimensionCount;
    _Field_size_(DimensionCount) const UINT* WindowSizes; // Size of the extracted patch
    _Field_size_(DimensionCount) const UINT* Strides; // Step size of the extracted patches
    _Field_size_(DimensionCount) const UINT* Dilations; // Dialations of the extracted patch
    _Field_size_(DimensionCount) const UINT* StartPadding; // Start padding of the "source tensor"
    _Field_size_(DimensionCount) const UINT* EndPadding; // End padding of the "source tensor"
};

#endif // DML_TARGET_VERSION >= 0x6400

// ===================================================================================================================
//   DML feature support queries
// ===================================================================================================================

#if DML_TARGET_VERSION >= 0x2000

enum DML_FEATURE_LEVEL
{
    DML_FEATURE_LEVEL_1_0 = 0x1000,
    DML_FEATURE_LEVEL_2_0 = 0x2000,
    DML_FEATURE_LEVEL_2_1 = 0x2100, 
    DML_FEATURE_LEVEL_3_0 = 0x3000,
    DML_FEATURE_LEVEL_3_1 = 0x3100,
    DML_FEATURE_LEVEL_4_0 = 0x4000,
    DML_FEATURE_LEVEL_4_1 = 0x4100,
    DML_FEATURE_LEVEL_5_0 = 0x5000,
    DML_FEATURE_LEVEL_5_1 = 0x5100,
    DML_FEATURE_LEVEL_5_2 = 0x5200,
    DML_FEATURE_LEVEL_6_0 = 0x6000,
    DML_FEATURE_LEVEL_6_1 = 0x6100,
    DML_FEATURE_LEVEL_6_2 = 0x6200,
    DML_FEATURE_LEVEL_6_3 = 0x6300,
    DML_FEATURE_LEVEL_6_4 = 0x6400,
};

#endif // DML_TARGET_VERSION >= 0x2000

enum DML_FEATURE
{
    DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT,

#if DML_TARGET_VERSION >= 0x2000
    DML_FEATURE_FEATURE_LEVELS,
#endif // DML_TARGET_VERSION >= 0x2000
};

struct DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT
{
    DML_TENSOR_DATA_TYPE DataType;
};

struct DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT
{
    BOOL IsSupported;
};

#if DML_TARGET_VERSION >= 0x2000

struct DML_FEATURE_QUERY_FEATURE_LEVELS
{
    UINT RequestedFeatureLevelCount;
    _Field_size_(RequestedFeatureLevelCount) const DML_FEATURE_LEVEL* RequestedFeatureLevels;
};

struct DML_FEATURE_DATA_FEATURE_LEVELS
{
    DML_FEATURE_LEVEL MaxSupportedFeatureLevel;
};

#endif // DML_TARGET_VERSION >= 0x2000

// ===================================================================================================================
//   DML device functions, enumerations, and structures
// ===================================================================================================================

struct DML_BINDING_TABLE_DESC
{
    IDMLDispatchable* Dispatchable;
    D3D12_CPU_DESCRIPTOR_HANDLE CPUDescriptorHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE GPUDescriptorHandle;
    UINT SizeInDescriptors;
};

enum DML_EXECUTION_FLAGS
{
    DML_EXECUTION_FLAG_NONE = 0,
    DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION = 0x1,
    DML_EXECUTION_FLAG_DISABLE_META_COMMANDS = 0x2,
    DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE = 0x4,
};

DEFINE_ENUM_FLAG_OPERATORS(DML_EXECUTION_FLAGS)

enum DML_CREATE_DEVICE_FLAGS
{
    DML_CREATE_DEVICE_FLAG_NONE = 0,
    DML_CREATE_DEVICE_FLAG_DEBUG = 0x1,
};

DEFINE_ENUM_FLAG_OPERATORS(DML_CREATE_DEVICE_FLAGS)

STDAPI DMLCreateDevice(
    ID3D12Device* d3d12Device,
    DML_CREATE_DEVICE_FLAGS flags,
    REFIID riid, // Expected: IDMLDevice
    _COM_Outptr_opt_ void** ppv
    );

#if DML_TARGET_VERSION >= 0x2000

STDAPI DMLCreateDevice1(
    ID3D12Device* d3d12Device,
    DML_CREATE_DEVICE_FLAGS flags,
    DML_FEATURE_LEVEL minimumFeatureLevel,
    REFIID riid, // Expected: IDMLDevice
    _COM_Outptr_opt_ void** ppv
    );

#endif // DML_TARGET_VERSION >= 0x2000

// ===================================================================================================================
//   DML object
// ===================================================================================================================

interface DML_DECLARE_INTERFACE("c8263aac-9e0c-4a2d-9b8e-007521a3317c") IDMLObject : IUnknown
{
    IFACEMETHOD(GetPrivateData)(
        REFGUID guid,
        _Inout_ UINT* dataSize,
        _Out_writes_bytes_opt_(*dataSize) void* data
        ) = 0;

    IFACEMETHOD(SetPrivateData)(
        REFGUID guid,
        UINT dataSize,
        _In_reads_bytes_opt_(dataSize) const void* data
        ) = 0;

    IFACEMETHOD(SetPrivateDataInterface)(
        REFGUID guid,
        _In_opt_ IUnknown* data
        ) = 0;

    IFACEMETHOD(SetName)(
        PCWSTR name
        ) = 0;
};

// ===================================================================================================================
//   DML device
// ===================================================================================================================

interface DML_DECLARE_INTERFACE("6dbd6437-96fd-423f-a98c-ae5e7c2a573f") IDMLDevice : IDMLObject
{
    IFACEMETHOD(CheckFeatureSupport)(
        DML_FEATURE feature,
        UINT featureQueryDataSize,
        _In_reads_bytes_opt_(featureQueryDataSize) const void* featureQueryData,
        UINT featureSupportDataSize,
        _Out_writes_bytes_(featureSupportDataSize) void* featureSupportData
        ) = 0;
    
    IFACEMETHOD(CreateOperator)(
        const DML_OPERATOR_DESC* desc,
        REFIID riid, // expected: IDMLOperator
        _COM_Outptr_opt_ void** ppv
        ) = 0;
    
    IFACEMETHOD(CompileOperator)(
        IDMLOperator* op,
        DML_EXECUTION_FLAGS flags,
        REFIID riid, // expected: IDMLCompiledOperator
        _COM_Outptr_opt_ void** ppv
        ) = 0;
    
    IFACEMETHOD(CreateOperatorInitializer)(
        UINT operatorCount,
        _In_reads_opt_(operatorCount) IDMLCompiledOperator* const* operators,
        REFIID riid, // expected: IDMLOperatorInitializer
        _COM_Outptr_ void** ppv
        ) = 0;
    
    IFACEMETHOD(CreateCommandRecorder)(
        REFIID riid, // expected: IDMLCommandRecorder
        _COM_Outptr_ void** ppv
        ) = 0;
    
    IFACEMETHOD(CreateBindingTable)(
        _In_opt_ const DML_BINDING_TABLE_DESC* desc,
        REFIID riid, // expected: IDMLBindingTable
        _COM_Outptr_ void** ppv
        ) = 0;
    
    IFACEMETHOD(Evict)(
        UINT count,
        _In_reads_(count) IDMLPageable* const* ppObjects
        ) = 0;
    
    IFACEMETHOD(MakeResident)(
        UINT count,
        _In_reads_(count) IDMLPageable* const* ppObjects
        ) = 0;
    
    IFACEMETHOD(GetDeviceRemovedReason)(
        ) = 0;

    IFACEMETHOD(GetParentDevice)(
        REFIID riid,
        _COM_Outptr_ void** ppv
        ) = 0;
};


// ===================================================================================================================
//   DML device children
// ===================================================================================================================

interface DML_DECLARE_INTERFACE("27e83142-8165-49e3-974e-2fd66e4cb69d") IDMLDeviceChild : IDMLObject
{
    IFACEMETHOD(GetDevice)(
        REFIID riid, // expected: IDMLDevice
        _COM_Outptr_ void** ppv
        ) = 0;
};

interface DML_DECLARE_INTERFACE("b1ab0825-4542-4a4b-8617-6dde6e8f6201") IDMLPageable : IDMLDeviceChild
{
};


// ===================================================================================================================
//   DML operator
// ===================================================================================================================

interface DML_DECLARE_INTERFACE("26caae7a-3081-4633-9581-226fbe57695d") IDMLOperator : IDMLDeviceChild
{
};


// ===================================================================================================================
//   DML dispatchable
// ===================================================================================================================

struct DML_BINDING_PROPERTIES
{
    UINT RequiredDescriptorCount;
    UINT64 TemporaryResourceSize;
    UINT64 PersistentResourceSize;
};

interface DML_DECLARE_INTERFACE("dcb821a8-1039-441e-9f1c-b1759c2f3cec") IDMLDispatchable : IDMLPageable
{
    IFACEMETHOD_(DML_BINDING_PROPERTIES, GetBindingProperties)() = 0;
};


// ===================================================================================================================
//   DML compiled operator
// ===================================================================================================================

interface DML_DECLARE_INTERFACE("6b15e56a-bf5c-4902-92d8-da3a650afea4") IDMLCompiledOperator : IDMLDispatchable
{
};


// ===================================================================================================================
//   DML operator initializer
// ===================================================================================================================

interface DML_DECLARE_INTERFACE("427c1113-435c-469c-8676-4d5dd072f813") IDMLOperatorInitializer : IDMLDispatchable
{
    IFACEMETHOD(Reset)(
        UINT operatorCount,
        _In_reads_opt_(operatorCount) IDMLCompiledOperator* const* operators
        ) = 0;
};

// ===================================================================================================================
//   DML binding table
// ===================================================================================================================

enum DML_BINDING_TYPE
{
    DML_BINDING_TYPE_NONE,
    DML_BINDING_TYPE_BUFFER,
    DML_BINDING_TYPE_BUFFER_ARRAY,
};

struct DML_BINDING_DESC
{
    DML_BINDING_TYPE Type;
    _Field_size_opt_(_Inexpressible_("Dependent on binding type")) const void* Desc;
};

struct DML_BUFFER_BINDING
{
    _Maybenull_ ID3D12Resource* Buffer;
    UINT64 Offset;
    UINT64 SizeInBytes;
};

struct DML_BUFFER_ARRAY_BINDING
{
    UINT BindingCount;
    _Field_size_(BindingCount) const DML_BUFFER_BINDING* Bindings;
};

interface DML_DECLARE_INTERFACE("29c687dc-de74-4e3b-ab00-1168f2fc3cfc") IDMLBindingTable : IDMLDeviceChild
{
    IFACEMETHOD_(void, BindInputs)(
        UINT bindingCount,
        _In_reads_opt_(bindingCount) const DML_BINDING_DESC* bindings
        ) = 0;

    IFACEMETHOD_(void, BindOutputs)(
        UINT bindingCount,
        _In_reads_opt_(bindingCount) const DML_BINDING_DESC* bindings
        ) = 0;

    IFACEMETHOD_(void, BindTemporaryResource)(
        _In_opt_ const DML_BINDING_DESC* binding
        ) = 0;

    IFACEMETHOD_(void, BindPersistentResource)(
        _In_opt_ const DML_BINDING_DESC* binding
        ) = 0;

    IFACEMETHOD(Reset)(
        _In_opt_ const DML_BINDING_TABLE_DESC* desc
        ) = 0;
};


// ===================================================================================================================
//   DML command recorder
// ===================================================================================================================

interface DML_DECLARE_INTERFACE("e6857a76-2e3e-4fdd-bff4-5d2ba10fb453") IDMLCommandRecorder : IDMLDeviceChild
{
    IFACEMETHOD_(void, RecordDispatch)(
        ID3D12CommandList* commandList,
        IDMLDispatchable* dispatchable,
        IDMLBindingTable* bindings
        ) = 0;
};


// ===================================================================================================================
//   DML debug
// ===================================================================================================================

interface DML_DECLARE_INTERFACE("7d6f3ac9-394a-4ac3-92a7-390cc57a8217") IDMLDebugDevice : IUnknown
{
    IFACEMETHOD_(void, SetMuteDebugOutput)(
        BOOL mute
        ) = 0;
};


// =================================================================================================================== 
// DML graph 
// =================================================================================================================== 

#if DML_TARGET_VERSION >= 0x2100

enum DML_GRAPH_EDGE_TYPE 
{ 
    DML_GRAPH_EDGE_TYPE_INVALID, 
    DML_GRAPH_EDGE_TYPE_INPUT, 
    DML_GRAPH_EDGE_TYPE_OUTPUT, 
    DML_GRAPH_EDGE_TYPE_INTERMEDIATE, 
}; 

struct DML_GRAPH_EDGE_DESC 
{ 
    DML_GRAPH_EDGE_TYPE Type; 
    _Field_size_(_Inexpressible_("Dependent on edge type")) const void* Desc; 
}; 

struct DML_INPUT_GRAPH_EDGE_DESC 
{ 
    UINT GraphInputIndex; 
    UINT ToNodeIndex; 
    UINT ToNodeInputIndex; 
    _Field_z_ _Maybenull_ const char* Name; 
}; 

struct DML_OUTPUT_GRAPH_EDGE_DESC 
{ 
    UINT FromNodeIndex; 
    UINT FromNodeOutputIndex; 
    UINT GraphOutputIndex; 
    _Field_z_ _Maybenull_ const char* Name; 
}; 

struct DML_INTERMEDIATE_GRAPH_EDGE_DESC 
{ 
    UINT FromNodeIndex; 
    UINT FromNodeOutputIndex; 
    UINT ToNodeIndex; 
    UINT ToNodeInputIndex; 
    _Field_z_ _Maybenull_ const char* Name; 
}; 

enum DML_GRAPH_NODE_TYPE 
{ 
    DML_GRAPH_NODE_TYPE_INVALID, 
    DML_GRAPH_NODE_TYPE_OPERATOR,
#if DML_TARGET_VERSION >= 0x6200
    DML_GRAPH_NODE_TYPE_CONSTANT
#endif // DML_TARGET_VERSION >= 0x6200
}; 

struct DML_GRAPH_NODE_DESC 
{ 
    DML_GRAPH_NODE_TYPE Type; 
    _Field_size_(_Inexpressible_("Dependent on node type")) const void* Desc; 
}; 

struct DML_OPERATOR_GRAPH_NODE_DESC 
{ 
    IDMLOperator* Operator; 
    _Field_z_ _Maybenull_ const char* Name; 
}; 

#if DML_TARGET_VERSION >= 0x6200
struct DML_CONSTANT_DATA_GRAPH_NODE_DESC 
{ 
    _Field_size_bytes_(DataSize) const void* Data;
    SIZE_T DataSize;
    _Field_z_ _Maybenull_ const char* Name; 
}; 
#endif // DML_TARGET_VERSION >= 0x6200

struct DML_GRAPH_DESC 
{ 
    UINT InputCount; 
    UINT OutputCount; 

    UINT NodeCount; 
    _Field_size_(NodeCount) const DML_GRAPH_NODE_DESC* Nodes; 

    UINT InputEdgeCount; 
    _Field_size_opt_(InputEdgeCount) const DML_GRAPH_EDGE_DESC* InputEdges; 

    UINT OutputEdgeCount; 
    _Field_size_(OutputEdgeCount) const DML_GRAPH_EDGE_DESC* OutputEdges; 

    UINT IntermediateEdgeCount; 
    _Field_size_opt_(IntermediateEdgeCount) const DML_GRAPH_EDGE_DESC* IntermediateEdges; 
}; 

interface DML_DECLARE_INTERFACE("a0884f9a-d2be-4355-aa5d-5901281ad1d2") IDMLDevice1 : IDMLDevice 
{ 
    IFACEMETHOD(CompileGraph)( 
        const DML_GRAPH_DESC* desc, 
        DML_EXECUTION_FLAGS flags, 
        REFIID riid, // expected: IDMLCompiledOperator 
        _COM_Outptr_opt_ void** ppv 
        ) = 0; 
};

#endif // DML_TARGET_VERSION >= 0x2100

#endif // WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES)
#endif // DIRECTML_H
