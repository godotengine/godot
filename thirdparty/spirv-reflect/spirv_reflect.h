/*
 Copyright 2017-2022 Google Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/*

VERSION HISTORY

  1.0   (2018-03-27) Initial public release

*/

/*!

 @file spirv_reflect.h

*/
#ifndef SPIRV_REFLECT_H
#define SPIRV_REFLECT_H

#if defined(SPIRV_REFLECT_USE_SYSTEM_SPIRV_H)
#include <spirv/unified1/spirv.h>
#else
#include "./include/spirv/unified1/spirv.h"
#endif


#include <stdint.h>
#include <string.h>

#ifdef _MSC_VER
  #define SPV_REFLECT_DEPRECATED(msg_str) __declspec(deprecated("This symbol is deprecated. Details: " msg_str))
#elif defined(__clang__)
  #define SPV_REFLECT_DEPRECATED(msg_str) __attribute__((deprecated(msg_str)))
#elif defined(__GNUC__)
  #if GCC_VERSION >= 40500
    #define SPV_REFLECT_DEPRECATED(msg_str) __attribute__((deprecated(msg_str)))
  #else
    #define SPV_REFLECT_DEPRECATED(msg_str) __attribute__((deprecated))
  #endif
#else
  #define SPV_REFLECT_DEPRECATED(msg_str)
#endif

/*! @enum SpvReflectResult

*/
typedef enum SpvReflectResult {
  SPV_REFLECT_RESULT_SUCCESS,
  SPV_REFLECT_RESULT_NOT_READY,
  SPV_REFLECT_RESULT_ERROR_PARSE_FAILED,
  SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED,
  SPV_REFLECT_RESULT_ERROR_RANGE_EXCEEDED,
  SPV_REFLECT_RESULT_ERROR_NULL_POINTER,
  SPV_REFLECT_RESULT_ERROR_INTERNAL_ERROR,
  SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH,
  SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND,
  SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_CODE_SIZE,
  SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_MAGIC_NUMBER,
  SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_EOF,
  SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE,
  SPV_REFLECT_RESULT_ERROR_SPIRV_SET_NUMBER_OVERFLOW,
  SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_STORAGE_CLASS,
  SPV_REFLECT_RESULT_ERROR_SPIRV_RECURSION,
  SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_INSTRUCTION,
  SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_BLOCK_DATA,
  SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_BLOCK_MEMBER_REFERENCE,
  SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ENTRY_POINT,
  SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_EXECUTION_MODE,
} SpvReflectResult;

/*! @enum SpvReflectModuleFlagBits

SPV_REFLECT_MODULE_FLAG_NO_COPY - Disables copying of SPIR-V code 
  when a SPIRV-Reflect shader module is created. It is the 
  responsibility of the calling program to ensure that the pointer
  remains valid and the memory it's pointing to is not freed while
  SPIRV-Reflect operations are taking place. Freeing the backing 
  memory will cause undefined behavior or most likely a crash.
  This is flag is intended for cases where the memory overhead of
  storing the copied SPIR-V is undesirable.

*/
typedef enum SpvReflectModuleFlagBits {
  SPV_REFLECT_MODULE_FLAG_NONE    = 0x00000000,
  SPV_REFLECT_MODULE_FLAG_NO_COPY = 0x00000001,
} SpvReflectModuleFlagBits;

typedef uint32_t SpvReflectModuleFlags;

/*! @enum SpvReflectTypeFlagBits

*/
typedef enum SpvReflectTypeFlagBits {
  SPV_REFLECT_TYPE_FLAG_UNDEFINED                       = 0x00000000,
  SPV_REFLECT_TYPE_FLAG_VOID                            = 0x00000001,
  SPV_REFLECT_TYPE_FLAG_BOOL                            = 0x00000002,
  SPV_REFLECT_TYPE_FLAG_INT                             = 0x00000004,
  SPV_REFLECT_TYPE_FLAG_FLOAT                           = 0x00000008,
  SPV_REFLECT_TYPE_FLAG_VECTOR                          = 0x00000100,
  SPV_REFLECT_TYPE_FLAG_MATRIX                          = 0x00000200,
  SPV_REFLECT_TYPE_FLAG_EXTERNAL_IMAGE                  = 0x00010000,
  SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLER                = 0x00020000,
  SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLED_IMAGE          = 0x00040000,
  SPV_REFLECT_TYPE_FLAG_EXTERNAL_BLOCK                  = 0x00080000,
  SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE = 0x00100000,
  SPV_REFLECT_TYPE_FLAG_EXTERNAL_MASK                   = 0x00FF0000,
  SPV_REFLECT_TYPE_FLAG_STRUCT                          = 0x10000000,
  SPV_REFLECT_TYPE_FLAG_ARRAY                           = 0x20000000,
} SpvReflectTypeFlagBits;

typedef uint32_t SpvReflectTypeFlags;

/*! @enum SpvReflectDecorationBits

NOTE: HLSL row_major and column_major decorations are reversed
      in SPIR-V. Meaning that matrices declrations with row_major
      will get reflected as column_major and vice versa. The
      row and column decorations get appied during the compilation.
      SPIRV-Reflect reads the data as is and does not make any
      attempt to correct it to match what's in the source.

*/
typedef enum SpvReflectDecorationFlagBits {
  SPV_REFLECT_DECORATION_NONE                   = 0x00000000,
  SPV_REFLECT_DECORATION_BLOCK                  = 0x00000001,
  SPV_REFLECT_DECORATION_BUFFER_BLOCK           = 0x00000002,
  SPV_REFLECT_DECORATION_ROW_MAJOR              = 0x00000004,
  SPV_REFLECT_DECORATION_COLUMN_MAJOR           = 0x00000008,
  SPV_REFLECT_DECORATION_BUILT_IN               = 0x00000010,
  SPV_REFLECT_DECORATION_NOPERSPECTIVE          = 0x00000020,
  SPV_REFLECT_DECORATION_FLAT                   = 0x00000040,
  SPV_REFLECT_DECORATION_NON_WRITABLE           = 0x00000080,
  SPV_REFLECT_DECORATION_RELAXED_PRECISION      = 0x00000100,
  SPV_REFLECT_DECORATION_NON_READABLE           = 0x00000200,
} SpvReflectDecorationFlagBits;

typedef uint32_t SpvReflectDecorationFlags;

/*! @enum SpvReflectResourceType

*/
typedef enum SpvReflectResourceType {
  SPV_REFLECT_RESOURCE_FLAG_UNDEFINED           = 0x00000000,
  SPV_REFLECT_RESOURCE_FLAG_SAMPLER             = 0x00000001,
  SPV_REFLECT_RESOURCE_FLAG_CBV                 = 0x00000002,
  SPV_REFLECT_RESOURCE_FLAG_SRV                 = 0x00000004,
  SPV_REFLECT_RESOURCE_FLAG_UAV                 = 0x00000008,
} SpvReflectResourceType;

/*! @enum SpvReflectFormat

*/
typedef enum SpvReflectFormat {
  SPV_REFLECT_FORMAT_UNDEFINED           =   0, // = VK_FORMAT_UNDEFINED
  SPV_REFLECT_FORMAT_R32_UINT            =  98, // = VK_FORMAT_R32_UINT
  SPV_REFLECT_FORMAT_R32_SINT            =  99, // = VK_FORMAT_R32_SINT
  SPV_REFLECT_FORMAT_R32_SFLOAT          = 100, // = VK_FORMAT_R32_SFLOAT
  SPV_REFLECT_FORMAT_R32G32_UINT         = 101, // = VK_FORMAT_R32G32_UINT
  SPV_REFLECT_FORMAT_R32G32_SINT         = 102, // = VK_FORMAT_R32G32_SINT
  SPV_REFLECT_FORMAT_R32G32_SFLOAT       = 103, // = VK_FORMAT_R32G32_SFLOAT
  SPV_REFLECT_FORMAT_R32G32B32_UINT      = 104, // = VK_FORMAT_R32G32B32_UINT
  SPV_REFLECT_FORMAT_R32G32B32_SINT      = 105, // = VK_FORMAT_R32G32B32_SINT
  SPV_REFLECT_FORMAT_R32G32B32_SFLOAT    = 106, // = VK_FORMAT_R32G32B32_SFLOAT
  SPV_REFLECT_FORMAT_R32G32B32A32_UINT   = 107, // = VK_FORMAT_R32G32B32A32_UINT
  SPV_REFLECT_FORMAT_R32G32B32A32_SINT   = 108, // = VK_FORMAT_R32G32B32A32_SINT
  SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT = 109, // = VK_FORMAT_R32G32B32A32_SFLOAT
  SPV_REFLECT_FORMAT_R64_UINT            = 110, // = VK_FORMAT_R64_UINT
  SPV_REFLECT_FORMAT_R64_SINT            = 111, // = VK_FORMAT_R64_SINT
  SPV_REFLECT_FORMAT_R64_SFLOAT          = 112, // = VK_FORMAT_R64_SFLOAT
  SPV_REFLECT_FORMAT_R64G64_UINT         = 113, // = VK_FORMAT_R64G64_UINT
  SPV_REFLECT_FORMAT_R64G64_SINT         = 114, // = VK_FORMAT_R64G64_SINT
  SPV_REFLECT_FORMAT_R64G64_SFLOAT       = 115, // = VK_FORMAT_R64G64_SFLOAT
  SPV_REFLECT_FORMAT_R64G64B64_UINT      = 116, // = VK_FORMAT_R64G64B64_UINT
  SPV_REFLECT_FORMAT_R64G64B64_SINT      = 117, // = VK_FORMAT_R64G64B64_SINT
  SPV_REFLECT_FORMAT_R64G64B64_SFLOAT    = 118, // = VK_FORMAT_R64G64B64_SFLOAT
  SPV_REFLECT_FORMAT_R64G64B64A64_UINT   = 119, // = VK_FORMAT_R64G64B64A64_UINT
  SPV_REFLECT_FORMAT_R64G64B64A64_SINT   = 120, // = VK_FORMAT_R64G64B64A64_SINT
  SPV_REFLECT_FORMAT_R64G64B64A64_SFLOAT = 121, // = VK_FORMAT_R64G64B64A64_SFLOAT
} SpvReflectFormat;

/*! @enum SpvReflectVariableFlagBits

*/
enum SpvReflectVariableFlagBits{
  SPV_REFLECT_VARIABLE_FLAGS_NONE   = 0x00000000,
  SPV_REFLECT_VARIABLE_FLAGS_UNUSED = 0x00000001,
};

typedef uint32_t SpvReflectVariableFlags;

/*! @enum SpvReflectDescriptorType

*/
typedef enum SpvReflectDescriptorType {
  SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER                    =  0,        // = VK_DESCRIPTOR_TYPE_SAMPLER
  SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER     =  1,        // = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
  SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE              =  2,        // = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
  SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE              =  3,        // = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
  SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER       =  4,        // = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
  SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER       =  5,        // = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
  SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER             =  6,        // = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
  SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER             =  7,        // = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
  SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC     =  8,        // = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
  SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC     =  9,        // = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
  SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT           = 10,        // = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT
  SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR = 1000150000 // = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
} SpvReflectDescriptorType;

/*! @enum SpvReflectShaderStageFlagBits

*/
typedef enum SpvReflectShaderStageFlagBits {
  SPV_REFLECT_SHADER_STAGE_VERTEX_BIT                  = 0x00000001, // = VK_SHADER_STAGE_VERTEX_BIT
  SPV_REFLECT_SHADER_STAGE_TESSELLATION_CONTROL_BIT    = 0x00000002, // = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
  SPV_REFLECT_SHADER_STAGE_TESSELLATION_EVALUATION_BIT = 0x00000004, // = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
  SPV_REFLECT_SHADER_STAGE_GEOMETRY_BIT                = 0x00000008, // = VK_SHADER_STAGE_GEOMETRY_BIT
  SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT                = 0x00000010, // = VK_SHADER_STAGE_FRAGMENT_BIT
  SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT                 = 0x00000020, // = VK_SHADER_STAGE_COMPUTE_BIT
  SPV_REFLECT_SHADER_STAGE_TASK_BIT_NV                 = 0x00000040, // = VK_SHADER_STAGE_TASK_BIT_NV
  SPV_REFLECT_SHADER_STAGE_MESH_BIT_NV                 = 0x00000080, // = VK_SHADER_STAGE_MESH_BIT_NV
  SPV_REFLECT_SHADER_STAGE_RAYGEN_BIT_KHR              = 0x00000100, // = VK_SHADER_STAGE_RAYGEN_BIT_KHR
  SPV_REFLECT_SHADER_STAGE_ANY_HIT_BIT_KHR             = 0x00000200, // = VK_SHADER_STAGE_ANY_HIT_BIT_KHR
  SPV_REFLECT_SHADER_STAGE_CLOSEST_HIT_BIT_KHR         = 0x00000400, // = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
  SPV_REFLECT_SHADER_STAGE_MISS_BIT_KHR                = 0x00000800, // = VK_SHADER_STAGE_MISS_BIT_KHR
  SPV_REFLECT_SHADER_STAGE_INTERSECTION_BIT_KHR        = 0x00001000, // = VK_SHADER_STAGE_INTERSECTION_BIT_KHR
  SPV_REFLECT_SHADER_STAGE_CALLABLE_BIT_KHR            = 0x00002000, // = VK_SHADER_STAGE_CALLABLE_BIT_KHR

} SpvReflectShaderStageFlagBits;

/*! @enum SpvReflectGenerator

*/
typedef enum SpvReflectGenerator {
  SPV_REFLECT_GENERATOR_KHRONOS_LLVM_SPIRV_TRANSLATOR         = 6,
  SPV_REFLECT_GENERATOR_KHRONOS_SPIRV_TOOLS_ASSEMBLER         = 7,
  SPV_REFLECT_GENERATOR_KHRONOS_GLSLANG_REFERENCE_FRONT_END   = 8,
  SPV_REFLECT_GENERATOR_GOOGLE_SHADERC_OVER_GLSLANG           = 13,
  SPV_REFLECT_GENERATOR_GOOGLE_SPIREGG                        = 14,
  SPV_REFLECT_GENERATOR_GOOGLE_RSPIRV                         = 15,
  SPV_REFLECT_GENERATOR_X_LEGEND_MESA_MESAIR_SPIRV_TRANSLATOR = 16,
  SPV_REFLECT_GENERATOR_KHRONOS_SPIRV_TOOLS_LINKER            = 17,
  SPV_REFLECT_GENERATOR_WINE_VKD3D_SHADER_COMPILER            = 18,
  SPV_REFLECT_GENERATOR_CLAY_CLAY_SHADER_COMPILER             = 19,
} SpvReflectGenerator;

enum {
  SPV_REFLECT_MAX_ARRAY_DIMS                    = 32,
  SPV_REFLECT_MAX_DESCRIPTOR_SETS               = 64,
};

enum {
  SPV_REFLECT_BINDING_NUMBER_DONT_CHANGE        = ~0,
  SPV_REFLECT_SET_NUMBER_DONT_CHANGE            = ~0
};

typedef struct SpvReflectNumericTraits {
  struct Scalar {
    uint32_t                        width;
    uint32_t                        signedness;
  } scalar;

  struct Vector {
    uint32_t                        component_count;
  } vector;

  struct Matrix {
    uint32_t                        column_count;
    uint32_t                        row_count;
    uint32_t                        stride; // Measured in bytes
  } matrix;
} SpvReflectNumericTraits;

typedef struct SpvReflectImageTraits {
  SpvDim                            dim;
  uint32_t                          depth;
  uint32_t                          arrayed;
  uint32_t                          ms; // 0: single-sampled; 1: multisampled
  uint32_t                          sampled;
  SpvImageFormat                    image_format;
} SpvReflectImageTraits;

typedef struct SpvReflectArrayTraits {
  uint32_t                          dims_count;
  // Each entry is: 0xFFFFFFFF for a specialization constant dimension,
  // 0 for a runtime array dimension, and the array length otherwise.
  uint32_t                          dims[SPV_REFLECT_MAX_ARRAY_DIMS];
  // Stores Ids for dimensions that are specialization constants
  uint32_t                          spec_constant_op_ids[SPV_REFLECT_MAX_ARRAY_DIMS];
  uint32_t                          stride; // Measured in bytes
} SpvReflectArrayTraits;

typedef struct SpvReflectBindingArrayTraits {
  uint32_t                          dims_count;
  uint32_t                          dims[SPV_REFLECT_MAX_ARRAY_DIMS];
} SpvReflectBindingArrayTraits;

/*! @struct SpvReflectTypeDescription

*/
typedef struct SpvReflectTypeDescription {
  uint32_t                          id;
  SpvOp                             op;
  const char*                       type_name;
  const char*                       struct_member_name;
  SpvStorageClass                   storage_class;
  SpvReflectTypeFlags               type_flags;
  SpvReflectDecorationFlags         decoration_flags;

  struct Traits {
    SpvReflectNumericTraits         numeric;
    SpvReflectImageTraits           image;
    SpvReflectArrayTraits           array;
  } traits;

  uint32_t                          member_count;
  struct SpvReflectTypeDescription* members;
} SpvReflectTypeDescription;

// -- GODOT begin --
/*! @struct SpvReflectSpecializationConstant

*/

typedef enum SpvReflectSpecializationConstantType {
  SPV_REFLECT_SPECIALIZATION_CONSTANT_BOOL = 0,
  SPV_REFLECT_SPECIALIZATION_CONSTANT_INT = 1,
  SPV_REFLECT_SPECIALIZATION_CONSTANT_FLOAT = 2,
} SpvReflectSpecializationConstantType;

typedef struct SpvReflectSpecializationConstant {
  const char* name;
  uint32_t spirv_id;
  uint32_t constant_id;
  SpvReflectSpecializationConstantType constant_type;
  union {
    float float_value;
    uint32_t int_bool_value;
  } default_value;
} SpvReflectSpecializationConstant;
// -- GODOT end --

/*! @struct SpvReflectInterfaceVariable

*/
typedef struct SpvReflectInterfaceVariable {
  uint32_t                            spirv_id;
  const char*                         name;
  uint32_t                            location;
  SpvStorageClass                     storage_class;
  const char*                         semantic;
  SpvReflectDecorationFlags           decoration_flags;
  SpvBuiltIn                          built_in;
  SpvReflectNumericTraits             numeric;
  SpvReflectArrayTraits               array;

  uint32_t                            member_count;
  struct SpvReflectInterfaceVariable* members;

  SpvReflectFormat                    format;

  // NOTE: SPIR-V shares type references for variables
  //       that have the same underlying type. This means
  //       that the same type name will appear for multiple
  //       variables.
  SpvReflectTypeDescription*          type_description;

  struct {
    uint32_t                          location;
  } word_offset;
} SpvReflectInterfaceVariable;

/*! @struct SpvReflectBlockVariable

*/
typedef struct SpvReflectBlockVariable {
  uint32_t                          spirv_id;
  const char*                       name;
  uint32_t                          offset;           // Measured in bytes
  uint32_t                          absolute_offset;  // Measured in bytes
  uint32_t                          size;             // Measured in bytes
  uint32_t                          padded_size;      // Measured in bytes
  SpvReflectDecorationFlags         decoration_flags;
  SpvReflectNumericTraits           numeric;
  SpvReflectArrayTraits             array;
  SpvReflectVariableFlags           flags;

  uint32_t                          member_count;
  struct SpvReflectBlockVariable*   members;

  SpvReflectTypeDescription*        type_description;
} SpvReflectBlockVariable;

/*! @struct SpvReflectDescriptorBinding

*/
typedef struct SpvReflectDescriptorBinding {
  uint32_t                            spirv_id;
  const char*                         name;
  uint32_t                            binding;
  uint32_t                            input_attachment_index;
  uint32_t                            set;
  SpvReflectDescriptorType            descriptor_type;
  SpvReflectResourceType              resource_type;
  SpvReflectImageTraits               image;
  SpvReflectBlockVariable             block;
  SpvReflectBindingArrayTraits        array;
  uint32_t                            count;
  uint32_t                            accessed;
  uint32_t                            uav_counter_id;
  struct SpvReflectDescriptorBinding* uav_counter_binding;

  SpvReflectTypeDescription*          type_description;

  struct {
    uint32_t                          binding;
    uint32_t                          set;
  } word_offset;

  SpvReflectDecorationFlags           decoration_flags;
} SpvReflectDescriptorBinding;

/*! @struct SpvReflectDescriptorSet

*/
typedef struct SpvReflectDescriptorSet {
  uint32_t                          set;
  uint32_t                          binding_count;
  SpvReflectDescriptorBinding**     bindings;
} SpvReflectDescriptorSet;

/*! @struct SpvReflectEntryPoint

 */
typedef struct SpvReflectEntryPoint {
  const char*                       name;
  uint32_t                          id;

  SpvExecutionModel                 spirv_execution_model;
  SpvReflectShaderStageFlagBits     shader_stage;

  uint32_t                          input_variable_count;  
  SpvReflectInterfaceVariable**     input_variables;       
  uint32_t                          output_variable_count; 
  SpvReflectInterfaceVariable**     output_variables;      
  uint32_t                          interface_variable_count;
  SpvReflectInterfaceVariable*      interface_variables;

  uint32_t                          descriptor_set_count;
  SpvReflectDescriptorSet*          descriptor_sets;

  uint32_t                          used_uniform_count;
  uint32_t*                         used_uniforms;
  uint32_t                          used_push_constant_count;
  uint32_t*                         used_push_constants;

  uint32_t                          execution_mode_count;
  SpvExecutionMode*                 execution_modes;

  struct LocalSize {
    uint32_t                        x;
    uint32_t                        y;
    uint32_t                        z;
  } local_size;
  uint32_t                          invocations; // valid for geometry
  uint32_t                          output_vertices; // valid for geometry, tesselation
} SpvReflectEntryPoint;

/*! @struct SpvReflectCapability

*/
typedef struct SpvReflectCapability {
  SpvCapability                     value;
  uint32_t                          word_offset;
} SpvReflectCapability;

/*! @struct SpvReflectShaderModule

*/
typedef struct SpvReflectShaderModule {
  SpvReflectGenerator               generator;
  const char*                       entry_point_name;
  uint32_t                          entry_point_id;
  uint32_t                          entry_point_count;
  SpvReflectEntryPoint*             entry_points;
  SpvSourceLanguage                 source_language;
  uint32_t                          source_language_version;
  const char*                       source_file;
  const char*                       source_source;
  uint32_t                          capability_count;
  SpvReflectCapability*             capabilities;
  SpvExecutionModel                 spirv_execution_model;                            // Uses value(s) from first entry point
  SpvReflectShaderStageFlagBits     shader_stage;                                     // Uses value(s) from first entry point
  uint32_t                          descriptor_binding_count;                         // Uses value(s) from first entry point
  SpvReflectDescriptorBinding*      descriptor_bindings;                              // Uses value(s) from first entry point
  uint32_t                          descriptor_set_count;                             // Uses value(s) from first entry point
  SpvReflectDescriptorSet           descriptor_sets[SPV_REFLECT_MAX_DESCRIPTOR_SETS]; // Uses value(s) from first entry point
  uint32_t                          input_variable_count;                             // Uses value(s) from first entry point
  SpvReflectInterfaceVariable**     input_variables;                                  // Uses value(s) from first entry point
  uint32_t                          output_variable_count;                            // Uses value(s) from first entry point
  SpvReflectInterfaceVariable**     output_variables;                                 // Uses value(s) from first entry point
  uint32_t                          interface_variable_count;                         // Uses value(s) from first entry point
  SpvReflectInterfaceVariable*      interface_variables;                              // Uses value(s) from first entry point
  uint32_t                          push_constant_block_count;                        // Uses value(s) from first entry point
  SpvReflectBlockVariable*          push_constant_blocks;                             // Uses value(s) from first entry point
  // -- GODOT begin --
  uint32_t                          specialization_constant_count;
  SpvReflectSpecializationConstant* specialization_constants;
  // -- GODOT end --

  struct Internal {
    SpvReflectModuleFlags           module_flags;
    size_t                          spirv_size;
    uint32_t*                       spirv_code;
    uint32_t                        spirv_word_count;

    size_t                          type_description_count;
    SpvReflectTypeDescription*      type_descriptions;
  } * _internal;

} SpvReflectShaderModule;

#if defined(__cplusplus)
extern "C" {
#endif

/*! @fn spvReflectCreateShaderModule

 @param  size      Size in bytes of SPIR-V code.
 @param  p_code    Pointer to SPIR-V code.
 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @return           SPV_REFLECT_RESULT_SUCCESS on success.

*/
SpvReflectResult spvReflectCreateShaderModule(
  size_t                   size,
  const void*              p_code,
  SpvReflectShaderModule*  p_module
);

/*! @fn spvReflectCreateShaderModule2

 @param  flags     Flags for module creations.
 @param  size      Size in bytes of SPIR-V code.
 @param  p_code    Pointer to SPIR-V code.
 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @return           SPV_REFLECT_RESULT_SUCCESS on success.

*/
SpvReflectResult spvReflectCreateShaderModule2(
  SpvReflectModuleFlags    flags,
  size_t                   size,
  const void*              p_code,
  SpvReflectShaderModule*  p_module
);

SPV_REFLECT_DEPRECATED("renamed to spvReflectCreateShaderModule")
SpvReflectResult spvReflectGetShaderModule(
  size_t                   size,
  const void*              p_code,
  SpvReflectShaderModule*  p_module
);


/*! @fn spvReflectDestroyShaderModule

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.

*/
void spvReflectDestroyShaderModule(SpvReflectShaderModule* p_module);


/*! @fn spvReflectGetCodeSize

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @return           Returns the size of the SPIR-V in bytes

*/
uint32_t spvReflectGetCodeSize(const SpvReflectShaderModule* p_module);


/*! @fn spvReflectGetCode

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @return           Returns a const pointer to the compiled SPIR-V bytecode.

*/
const uint32_t* spvReflectGetCode(const SpvReflectShaderModule* p_module);

/*! @fn spvReflectGetEntryPoint

 @param  p_module     Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point  Name of the requested entry point.
 @return              Returns a const pointer to the requested entry point,
                      or NULL if it's not found.
*/
const SpvReflectEntryPoint* spvReflectGetEntryPoint(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point
);

/*! @fn spvReflectEnumerateDescriptorBindings

 @param  p_module     Pointer to an instance of SpvReflectShaderModule.
 @param  p_count      If pp_bindings is NULL, the module's descriptor binding
                      count (across all descriptor sets) will be stored here.
                      If pp_bindings is not NULL, *p_count must contain the
                      module's descriptor binding count.
 @param  pp_bindings  If NULL, the module's total descriptor binding count
                      will be written to *p_count.
                      If non-NULL, pp_bindings must point to an array with
                      *p_count entries, where pointers to the module's
                      descriptor bindings will be written. The caller must not
                      free the binding pointers written to this array.
 @return              If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                      Otherwise, the error code indicates the cause of the
                      failure.

*/
SpvReflectResult spvReflectEnumerateDescriptorBindings(
  const SpvReflectShaderModule*  p_module,
  uint32_t*                      p_count,
  SpvReflectDescriptorBinding**  pp_bindings
);

/*! @fn spvReflectEnumerateEntryPointDescriptorBindings
 @brief  Creates a listing of all descriptor bindings that are used in the
         static call tree of the given entry point.
 @param  p_module     Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point  The name of the entry point to get the descriptor bindings for.
 @param  p_count      If pp_bindings is NULL, the entry point's descriptor binding
                      count (across all descriptor sets) will be stored here.
                      If pp_bindings is not NULL, *p_count must contain the
                      entry points's descriptor binding count.
 @param  pp_bindings  If NULL, the entry point's total descriptor binding count
                      will be written to *p_count.
                      If non-NULL, pp_bindings must point to an array with
                      *p_count entries, where pointers to the entry point's
                      descriptor bindings will be written. The caller must not
                      free the binding pointers written to this array.
 @return              If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                      Otherwise, the error code indicates the cause of the
                      failure.

*/
SpvReflectResult spvReflectEnumerateEntryPointDescriptorBindings(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectDescriptorBinding** pp_bindings
);

/*! @fn spvReflectEnumerateDescriptorSets

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @param  p_count   If pp_sets is NULL, the module's descriptor set
                   count will be stored here.
                   If pp_sets is not NULL, *p_count must contain the
                   module's descriptor set count.
 @param  pp_sets   If NULL, the module's total descriptor set count
                   will be written to *p_count.
                   If non-NULL, pp_sets must point to an array with
                   *p_count entries, where pointers to the module's
                   descriptor sets will be written. The caller must not
                   free the descriptor set pointers written to this array.
 @return           If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                   Otherwise, the error code indicates the cause of the
                   failure.

*/
SpvReflectResult spvReflectEnumerateDescriptorSets(
  const SpvReflectShaderModule* p_module,
  uint32_t*                     p_count,
  SpvReflectDescriptorSet**     pp_sets
);

/*! @fn spvReflectEnumerateEntryPointDescriptorSets
 @brief  Creates a listing of all descriptor sets and their bindings that are
         used in the static call tree of a given entry point.
 @param  p_module    Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point The name of the entry point to get the descriptor bindings for.
 @param  p_count     If pp_sets is NULL, the module's descriptor set
                     count will be stored here.
                     If pp_sets is not NULL, *p_count must contain the
                     module's descriptor set count.
 @param  pp_sets     If NULL, the module's total descriptor set count
                     will be written to *p_count.
                     If non-NULL, pp_sets must point to an array with
                     *p_count entries, where pointers to the module's
                     descriptor sets will be written. The caller must not
                     free the descriptor set pointers written to this array.
 @return             If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                     Otherwise, the error code indicates the cause of the
                     failure.

*/
SpvReflectResult spvReflectEnumerateEntryPointDescriptorSets(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectDescriptorSet**     pp_sets
);


/*! @fn spvReflectEnumerateInterfaceVariables
 @brief  If the module contains multiple entry points, this will only get
         the interface variables for the first one.
 @param  p_module      Pointer to an instance of SpvReflectShaderModule.
 @param  p_count       If pp_variables is NULL, the module's interface variable
                       count will be stored here.
                       If pp_variables is not NULL, *p_count must contain
                       the module's interface variable count.
 @param  pp_variables  If NULL, the module's interface variable count will be
                       written to *p_count.
                       If non-NULL, pp_variables must point to an array with
                       *p_count entries, where pointers to the module's
                       interface variables will be written. The caller must not
                       free the interface variables written to this array.
 @return               If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                       Otherwise, the error code indicates the cause of the
                       failure.

*/
SpvReflectResult spvReflectEnumerateInterfaceVariables(
  const SpvReflectShaderModule* p_module,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
);

/*! @fn spvReflectEnumerateEntryPointInterfaceVariables
 @brief  Enumerate the interface variables for a given entry point.
 @param  entry_point The name of the entry point to get the interface variables for.
 @param  p_module      Pointer to an instance of SpvReflectShaderModule.
 @param  p_count       If pp_variables is NULL, the entry point's interface variable
                       count will be stored here.
                       If pp_variables is not NULL, *p_count must contain
                       the entry point's interface variable count.
 @param  pp_variables  If NULL, the entry point's interface variable count will be
                       written to *p_count.
                       If non-NULL, pp_variables must point to an array with
                       *p_count entries, where pointers to the entry point's
                       interface variables will be written. The caller must not
                       free the interface variables written to this array.
 @return               If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                       Otherwise, the error code indicates the cause of the
                       failure.

*/
SpvReflectResult spvReflectEnumerateEntryPointInterfaceVariables(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
);


/*! @fn spvReflectEnumerateInputVariables
 @brief  If the module contains multiple entry points, this will only get
         the input variables for the first one.
 @param  p_module      Pointer to an instance of SpvReflectShaderModule.
 @param  p_count       If pp_variables is NULL, the module's input variable
                       count will be stored here.
                       If pp_variables is not NULL, *p_count must contain
                       the module's input variable count.
 @param  pp_variables  If NULL, the module's input variable count will be
                       written to *p_count.
                       If non-NULL, pp_variables must point to an array with
                       *p_count entries, where pointers to the module's
                       input variables will be written. The caller must not
                       free the interface variables written to this array.
 @return               If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                       Otherwise, the error code indicates the cause of the
                       failure.

*/
SpvReflectResult spvReflectEnumerateInputVariables(
  const SpvReflectShaderModule* p_module,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
);

// -- GOODT begin --
/*! @fn spvReflectEnumerateSpecializationConstants
 @brief  If the module contains multiple entry points, this will only get
         the specialization constants for the first one.
 @param  p_module      Pointer to an instance of SpvReflectShaderModule.
 @param  p_count       If pp_constants is NULL, the module's specialization constant
                       count will be stored here.
                       If pp_variables is not NULL, *p_count must contain
                       the module's specialization constant count.
 @param  pp_variables  If NULL, the module's specialization constant count will be
                       written to *p_count.
                       If non-NULL, pp_constants must point to an array with
                       *p_count entries, where pointers to the module's
                       specialization constants will be written. The caller must not
                       free the specialization constants written to this array.
 @return               If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                       Otherwise, the error code indicates the cause of the
                       failure.

*/
SpvReflectResult spvReflectEnumerateSpecializationConstants(
  const SpvReflectShaderModule*      p_module,
  uint32_t*                          p_count,
  SpvReflectSpecializationConstant** pp_constants
);
// -- GODOT end --

/*! @fn spvReflectEnumerateEntryPointInputVariables
 @brief  Enumerate the input variables for a given entry point.
 @param  entry_point The name of the entry point to get the input variables for.
 @param  p_module      Pointer to an instance of SpvReflectShaderModule.
 @param  p_count       If pp_variables is NULL, the entry point's input variable
                       count will be stored here.
                       If pp_variables is not NULL, *p_count must contain
                       the entry point's input variable count.
 @param  pp_variables  If NULL, the entry point's input variable count will be
                       written to *p_count.
                       If non-NULL, pp_variables must point to an array with
                       *p_count entries, where pointers to the entry point's
                       input variables will be written. The caller must not
                       free the interface variables written to this array.
 @return               If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                       Otherwise, the error code indicates the cause of the
                       failure.

*/
SpvReflectResult spvReflectEnumerateEntryPointInputVariables(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
);


/*! @fn spvReflectEnumerateOutputVariables
 @brief  Note: If the module contains multiple entry points, this will only get
         the output variables for the first one.
 @param  p_module      Pointer to an instance of SpvReflectShaderModule.
 @param  p_count       If pp_variables is NULL, the module's output variable
                       count will be stored here.
                       If pp_variables is not NULL, *p_count must contain
                       the module's output variable count.
 @param  pp_variables  If NULL, the module's output variable count will be
                       written to *p_count.
                       If non-NULL, pp_variables must point to an array with
                       *p_count entries, where pointers to the module's
                       output variables will be written. The caller must not
                       free the interface variables written to this array.
 @return               If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                       Otherwise, the error code indicates the cause of the
                       failure.

*/
SpvReflectResult spvReflectEnumerateOutputVariables(
  const SpvReflectShaderModule* p_module,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
);

/*! @fn spvReflectEnumerateEntryPointOutputVariables
 @brief  Enumerate the output variables for a given entry point.
 @param  p_module      Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point   The name of the entry point to get the output variables for.
 @param  p_count       If pp_variables is NULL, the entry point's output variable
                       count will be stored here.
                       If pp_variables is not NULL, *p_count must contain
                       the entry point's output variable count.
 @param  pp_variables  If NULL, the entry point's output variable count will be
                       written to *p_count.
                       If non-NULL, pp_variables must point to an array with
                       *p_count entries, where pointers to the entry point's
                       output variables will be written. The caller must not
                       free the interface variables written to this array.
 @return               If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                       Otherwise, the error code indicates the cause of the
                       failure.

*/
SpvReflectResult spvReflectEnumerateEntryPointOutputVariables(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
);


/*! @fn spvReflectEnumeratePushConstantBlocks
 @brief  Note: If the module contains multiple entry points, this will only get
         the push constant blocks for the first one.
 @param  p_module   Pointer to an instance of SpvReflectShaderModule.
 @param  p_count    If pp_blocks is NULL, the module's push constant
                    block count will be stored here.
                    If pp_blocks is not NULL, *p_count must
                    contain the module's push constant block count.
 @param  pp_blocks  If NULL, the module's push constant block count
                    will be written to *p_count.
                    If non-NULL, pp_blocks must point to an
                    array with *p_count entries, where pointers to
                    the module's push constant blocks will be written.
                    The caller must not free the block variables written
                    to this array.
 @return            If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                    Otherwise, the error code indicates the cause of the
                    failure.

*/
SpvReflectResult spvReflectEnumeratePushConstantBlocks(
  const SpvReflectShaderModule* p_module,
  uint32_t*                     p_count,
  SpvReflectBlockVariable**     pp_blocks
);
SPV_REFLECT_DEPRECATED("renamed to spvReflectEnumeratePushConstantBlocks")
SpvReflectResult spvReflectEnumeratePushConstants(
  const SpvReflectShaderModule* p_module,
  uint32_t*                     p_count,
  SpvReflectBlockVariable**     pp_blocks
);

/*! @fn spvReflectEnumerateEntryPointPushConstantBlocks
 @brief  Enumerate the push constant blocks used in the static call tree of a
         given entry point.
 @param  p_module   Pointer to an instance of SpvReflectShaderModule.
 @param  p_count    If pp_blocks is NULL, the entry point's push constant
                    block count will be stored here.
                    If pp_blocks is not NULL, *p_count must
                    contain the entry point's push constant block count.
 @param  pp_blocks  If NULL, the entry point's push constant block count
                    will be written to *p_count.
                    If non-NULL, pp_blocks must point to an
                    array with *p_count entries, where pointers to
                    the entry point's push constant blocks will be written.
                    The caller must not free the block variables written
                    to this array.
 @return            If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                    Otherwise, the error code indicates the cause of the
                    failure.

*/
SpvReflectResult spvReflectEnumerateEntryPointPushConstantBlocks(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectBlockVariable**     pp_blocks
);


/*! @fn spvReflectGetDescriptorBinding

 @param  p_module        Pointer to an instance of SpvReflectShaderModule.
 @param  binding_number  The "binding" value of the requested descriptor
                         binding.
 @param  set_number      The "set" value of the requested descriptor binding.
 @param  p_result        If successful, SPV_REFLECT_RESULT_SUCCESS will be
                         written to *p_result. Otherwise, a error code
                         indicating the cause of the failure will be stored
                         here.
 @return                 If the module contains a descriptor binding that
                         matches the provided [binding_number, set_number]
                         values, a pointer to that binding is returned. The
                         caller must not free this pointer.
                         If no match can be found, or if an unrelated error
                         occurs, the return value will be NULL. Detailed
                         error results are written to *pResult.
@note                    If the module contains multiple desriptor bindings
                         with the same set and binding numbers, there are
                         no guarantees about which binding will be returned.

*/
const SpvReflectDescriptorBinding* spvReflectGetDescriptorBinding(
  const SpvReflectShaderModule* p_module,
  uint32_t                      binding_number,
  uint32_t                      set_number,
  SpvReflectResult*             p_result
);

/*! @fn spvReflectGetEntryPointDescriptorBinding
 @brief  Get the descriptor binding with the given binding number and set
         number that is used in the static call tree of a certain entry
         point.
 @param  p_module        Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point     The entry point to get the binding from.
 @param  binding_number  The "binding" value of the requested descriptor
                         binding.
 @param  set_number      The "set" value of the requested descriptor binding.
 @param  p_result        If successful, SPV_REFLECT_RESULT_SUCCESS will be
                         written to *p_result. Otherwise, a error code
                         indicating the cause of the failure will be stored
                         here.
 @return                 If the entry point contains a descriptor binding that
                         matches the provided [binding_number, set_number]
                         values, a pointer to that binding is returned. The
                         caller must not free this pointer.
                         If no match can be found, or if an unrelated error
                         occurs, the return value will be NULL. Detailed
                         error results are written to *pResult.
@note                    If the entry point contains multiple desriptor bindings
                         with the same set and binding numbers, there are
                         no guarantees about which binding will be returned.

*/
const SpvReflectDescriptorBinding* spvReflectGetEntryPointDescriptorBinding(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t                      binding_number,
  uint32_t                      set_number,
  SpvReflectResult*             p_result
);


/*! @fn spvReflectGetDescriptorSet

 @param  p_module    Pointer to an instance of SpvReflectShaderModule.
 @param  set_number  The "set" value of the requested descriptor set.
 @param  p_result    If successful, SPV_REFLECT_RESULT_SUCCESS will be
                     written to *p_result. Otherwise, a error code
                     indicating the cause of the failure will be stored
                     here.
 @return             If the module contains a descriptor set with the
                     provided set_number, a pointer to that set is
                     returned. The caller must not free this pointer.
                     If no match can be found, or if an unrelated error
                     occurs, the return value will be NULL. Detailed
                     error results are written to *pResult.

*/
const SpvReflectDescriptorSet* spvReflectGetDescriptorSet(
  const SpvReflectShaderModule* p_module,
  uint32_t                      set_number,
  SpvReflectResult*             p_result
);

/*! @fn spvReflectGetEntryPointDescriptorSet

 @param  p_module    Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point The entry point to get the descriptor set from.
 @param  set_number  The "set" value of the requested descriptor set.
 @param  p_result    If successful, SPV_REFLECT_RESULT_SUCCESS will be
                     written to *p_result. Otherwise, a error code
                     indicating the cause of the failure will be stored
                     here.
 @return             If the entry point contains a descriptor set with the
                     provided set_number, a pointer to that set is
                     returned. The caller must not free this pointer.
                     If no match can be found, or if an unrelated error
                     occurs, the return value will be NULL. Detailed
                     error results are written to *pResult.

*/
const SpvReflectDescriptorSet* spvReflectGetEntryPointDescriptorSet(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t                      set_number,
  SpvReflectResult*             p_result
);


/* @fn spvReflectGetInputVariableByLocation

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @param  location  The "location" value of the requested input variable.
                   A location of 0xFFFFFFFF will always return NULL
                   with *p_result == ELEMENT_NOT_FOUND.
 @param  p_result  If successful, SPV_REFLECT_RESULT_SUCCESS will be
                   written to *p_result. Otherwise, a error code
                   indicating the cause of the failure will be stored
                   here.
 @return           If the module contains an input interface variable
                   with the provided location value, a pointer to that
                   variable is returned. The caller must not free this
                   pointer.
                   If no match can be found, or if an unrelated error
                   occurs, the return value will be NULL. Detailed
                   error results are written to *pResult.
@note

*/
const SpvReflectInterfaceVariable* spvReflectGetInputVariableByLocation(
  const SpvReflectShaderModule* p_module,
  uint32_t                      location,
  SpvReflectResult*             p_result
);
SPV_REFLECT_DEPRECATED("renamed to spvReflectGetInputVariableByLocation")
const SpvReflectInterfaceVariable* spvReflectGetInputVariable(
  const SpvReflectShaderModule* p_module,
  uint32_t                      location,
  SpvReflectResult*             p_result
);

/* @fn spvReflectGetEntryPointInputVariableByLocation

 @param  p_module    Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point The entry point to get the input variable from.
 @param  location    The "location" value of the requested input variable.
                     A location of 0xFFFFFFFF will always return NULL
                     with *p_result == ELEMENT_NOT_FOUND.
 @param  p_result    If successful, SPV_REFLECT_RESULT_SUCCESS will be
                     written to *p_result. Otherwise, a error code
                     indicating the cause of the failure will be stored
                     here.
 @return             If the entry point contains an input interface variable
                     with the provided location value, a pointer to that
                     variable is returned. The caller must not free this
                     pointer.
                     If no match can be found, or if an unrelated error
                     occurs, the return value will be NULL. Detailed
                     error results are written to *pResult.
@note

*/
const SpvReflectInterfaceVariable* spvReflectGetEntryPointInputVariableByLocation(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  uint32_t                      location,
  SpvReflectResult*             p_result
);

/* @fn spvReflectGetInputVariableBySemantic

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @param  semantic  The "semantic" value of the requested input variable.
                   A semantic of NULL will return NULL.
                   A semantic of "" will always return NULL with
                   *p_result == ELEMENT_NOT_FOUND.
 @param  p_result  If successful, SPV_REFLECT_RESULT_SUCCESS will be
                   written to *p_result. Otherwise, a error code
                   indicating the cause of the failure will be stored
                   here.
 @return           If the module contains an input interface variable
                   with the provided semantic, a pointer to that
                   variable is returned. The caller must not free this
                   pointer.
                   If no match can be found, or if an unrelated error
                   occurs, the return value will be NULL. Detailed
                   error results are written to *pResult.
@note

*/
const SpvReflectInterfaceVariable* spvReflectGetInputVariableBySemantic(
  const SpvReflectShaderModule* p_module,
  const char*                   semantic,
  SpvReflectResult*             p_result
);

/* @fn spvReflectGetEntryPointInputVariableBySemantic

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point The entry point to get the input variable from.
 @param  semantic  The "semantic" value of the requested input variable.
                   A semantic of NULL will return NULL.
                   A semantic of "" will always return NULL with
                   *p_result == ELEMENT_NOT_FOUND.
 @param  p_result  If successful, SPV_REFLECT_RESULT_SUCCESS will be
                   written to *p_result. Otherwise, a error code
                   indicating the cause of the failure will be stored
                   here.
 @return           If the entry point contains an input interface variable
                   with the provided semantic, a pointer to that
                   variable is returned. The caller must not free this
                   pointer.
                   If no match can be found, or if an unrelated error
                   occurs, the return value will be NULL. Detailed
                   error results are written to *pResult.
@note

*/
const SpvReflectInterfaceVariable* spvReflectGetEntryPointInputVariableBySemantic(
  const SpvReflectShaderModule* p_module,
  const char*                   entry_point,
  const char*                   semantic,
  SpvReflectResult*             p_result
);

/* @fn spvReflectGetOutputVariableByLocation

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @param  location  The "location" value of the requested output variable.
                   A location of 0xFFFFFFFF will always return NULL
                   with *p_result == ELEMENT_NOT_FOUND.
 @param  p_result  If successful, SPV_REFLECT_RESULT_SUCCESS will be
                   written to *p_result. Otherwise, a error code
                   indicating the cause of the failure will be stored
                   here.
 @return           If the module contains an output interface variable
                   with the provided location value, a pointer to that
                   variable is returned. The caller must not free this
                   pointer.
                   If no match can be found, or if an unrelated error
                   occurs, the return value will be NULL. Detailed
                   error results are written to *pResult.
@note

*/
const SpvReflectInterfaceVariable* spvReflectGetOutputVariableByLocation(
  const SpvReflectShaderModule*  p_module,
  uint32_t                       location,
  SpvReflectResult*              p_result
);
SPV_REFLECT_DEPRECATED("renamed to spvReflectGetOutputVariableByLocation")
const SpvReflectInterfaceVariable* spvReflectGetOutputVariable(
  const SpvReflectShaderModule*  p_module,
  uint32_t                       location,
  SpvReflectResult*              p_result
);

/* @fn spvReflectGetEntryPointOutputVariableByLocation

 @param  p_module     Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point  The entry point to get the output variable from.
 @param  location     The "location" value of the requested output variable.
                      A location of 0xFFFFFFFF will always return NULL
                      with *p_result == ELEMENT_NOT_FOUND.
 @param  p_result     If successful, SPV_REFLECT_RESULT_SUCCESS will be
                      written to *p_result. Otherwise, a error code
                      indicating the cause of the failure will be stored
                      here.
 @return              If the entry point contains an output interface variable
                      with the provided location value, a pointer to that
                      variable is returned. The caller must not free this
                      pointer.
                      If no match can be found, or if an unrelated error
                      occurs, the return value will be NULL. Detailed
                      error results are written to *pResult.
@note

*/
const SpvReflectInterfaceVariable* spvReflectGetEntryPointOutputVariableByLocation(
  const SpvReflectShaderModule*  p_module,
  const char*                    entry_point,
  uint32_t                       location,
  SpvReflectResult*              p_result
);

/* @fn spvReflectGetOutputVariableBySemantic

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @param  semantic  The "semantic" value of the requested output variable.
                   A semantic of NULL will return NULL.
                   A semantic of "" will always return NULL with
                   *p_result == ELEMENT_NOT_FOUND.
 @param  p_result  If successful, SPV_REFLECT_RESULT_SUCCESS will be
                   written to *p_result. Otherwise, a error code
                   indicating the cause of the failure will be stored
                   here.
 @return           If the module contains an output interface variable
                   with the provided semantic, a pointer to that
                   variable is returned. The caller must not free this
                   pointer.
                   If no match can be found, or if an unrelated error
                   occurs, the return value will be NULL. Detailed
                   error results are written to *pResult.
@note

*/
const SpvReflectInterfaceVariable* spvReflectGetOutputVariableBySemantic(
  const SpvReflectShaderModule*  p_module,
  const char*                    semantic,
  SpvReflectResult*              p_result
);

/* @fn spvReflectGetEntryPointOutputVariableBySemantic

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point  The entry point to get the output variable from.
 @param  semantic  The "semantic" value of the requested output variable.
                   A semantic of NULL will return NULL.
                   A semantic of "" will always return NULL with
                   *p_result == ELEMENT_NOT_FOUND.
 @param  p_result  If successful, SPV_REFLECT_RESULT_SUCCESS will be
                   written to *p_result. Otherwise, a error code
                   indicating the cause of the failure will be stored
                   here.
 @return           If the entry point contains an output interface variable
                   with the provided semantic, a pointer to that
                   variable is returned. The caller must not free this
                   pointer.
                   If no match can be found, or if an unrelated error
                   occurs, the return value will be NULL. Detailed
                   error results are written to *pResult.
@note

*/
const SpvReflectInterfaceVariable* spvReflectGetEntryPointOutputVariableBySemantic(
  const SpvReflectShaderModule*  p_module,
  const char*                    entry_point,
  const char*                    semantic,
  SpvReflectResult*              p_result
);

/*! @fn spvReflectGetPushConstantBlock

 @param  p_module  Pointer to an instance of SpvReflectShaderModule.
 @param  index     The index of the desired block within the module's
                   array of push constant blocks.
 @param  p_result  If successful, SPV_REFLECT_RESULT_SUCCESS will be
                   written to *p_result. Otherwise, a error code
                   indicating the cause of the failure will be stored
                   here.
 @return           If the provided index is within range, a pointer to
                   the corresponding push constant block is returned.
                   The caller must not free this pointer.
                   If no match can be found, or if an unrelated error
                   occurs, the return value will be NULL. Detailed
                   error results are written to *pResult.

*/
const SpvReflectBlockVariable* spvReflectGetPushConstantBlock(
  const SpvReflectShaderModule*  p_module,
  uint32_t                       index,
  SpvReflectResult*              p_result
);
SPV_REFLECT_DEPRECATED("renamed to spvReflectGetPushConstantBlock")
const SpvReflectBlockVariable* spvReflectGetPushConstant(
  const SpvReflectShaderModule*  p_module,
  uint32_t                       index,
  SpvReflectResult*              p_result
);

/*! @fn spvReflectGetEntryPointPushConstantBlock
 @brief  Get the push constant block corresponding to the given entry point.
         As by the Vulkan specification there can be no more than one push
         constant block used by a given entry point, so if there is one it will
         be returned, otherwise NULL will be returned.
 @param  p_module     Pointer to an instance of SpvReflectShaderModule.
 @param  entry_point  The entry point to get the push constant block from.
 @param  p_result     If successful, SPV_REFLECT_RESULT_SUCCESS will be
                      written to *p_result. Otherwise, a error code
                      indicating the cause of the failure will be stored
                      here.
 @return              If the provided index is within range, a pointer to
                      the corresponding push constant block is returned.
                      The caller must not free this pointer.
                      If no match can be found, or if an unrelated error
                      occurs, the return value will be NULL. Detailed
                      error results are written to *pResult.

*/
const SpvReflectBlockVariable* spvReflectGetEntryPointPushConstantBlock(
  const SpvReflectShaderModule*  p_module,
  const char*                    entry_point,
  SpvReflectResult*              p_result
);


/*! @fn spvReflectChangeDescriptorBindingNumbers
 @brief  Assign new set and/or binding numbers to a descriptor binding.
         In addition to updating the reflection data, this function modifies
         the underlying SPIR-V bytecode. The updated code can be retrieved
         with spvReflectGetCode().  If the binding is used in multiple
         entry points within the module, it will be changed in all of them.
 @param  p_module            Pointer to an instance of SpvReflectShaderModule.
 @param  p_binding           Pointer to the descriptor binding to modify.
 @param  new_binding_number  The new binding number to assign to the
                             provided descriptor binding.
                             To leave the binding number unchanged, pass
                             SPV_REFLECT_BINDING_NUMBER_DONT_CHANGE.
 @param  new_set_number      The new set number to assign to the
                             provided descriptor binding. Successfully changing
                             a descriptor binding's set number invalidates all
                             existing SpvReflectDescriptorBinding and
                             SpvReflectDescriptorSet pointers from this module.
                             To leave the set number unchanged, pass
                             SPV_REFLECT_SET_NUMBER_DONT_CHANGE.
 @return                     If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                             Otherwise, the error code indicates the cause of
                             the failure.
*/
SpvReflectResult spvReflectChangeDescriptorBindingNumbers(
  SpvReflectShaderModule*            p_module,
  const SpvReflectDescriptorBinding* p_binding,
  uint32_t                           new_binding_number,
  uint32_t                           new_set_number
);
SPV_REFLECT_DEPRECATED("Renamed to spvReflectChangeDescriptorBindingNumbers")
SpvReflectResult spvReflectChangeDescriptorBindingNumber(
  SpvReflectShaderModule*            p_module,
  const SpvReflectDescriptorBinding* p_descriptor_binding,
  uint32_t                           new_binding_number,
  uint32_t                           optional_new_set_number
);

/*! @fn spvReflectChangeDescriptorSetNumber
 @brief  Assign a new set number to an entire descriptor set (including
         all descriptor bindings in that set).
         In addition to updating the reflection data, this function modifies
         the underlying SPIR-V bytecode. The updated code can be retrieved
         with spvReflectGetCode().  If the descriptor set is used in
         multiple entry points within the module, it will be modified in all
         of them.
 @param  p_module        Pointer to an instance of SpvReflectShaderModule.
 @param  p_set           Pointer to the descriptor binding to modify.
 @param  new_set_number  The new set number to assign to the
                         provided descriptor set, and all its descriptor
                         bindings. Successfully changing a descriptor
                         binding's set number invalidates all existing
                         SpvReflectDescriptorBinding and
                         SpvReflectDescriptorSet pointers from this module.
                         To leave the set number unchanged, pass
                         SPV_REFLECT_SET_NUMBER_DONT_CHANGE.
 @return                 If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                         Otherwise, the error code indicates the cause of
                         the failure.
*/
SpvReflectResult spvReflectChangeDescriptorSetNumber(
  SpvReflectShaderModule*        p_module,
  const SpvReflectDescriptorSet* p_set,
  uint32_t                       new_set_number
);

/*! @fn spvReflectChangeInputVariableLocation
 @brief  Assign a new location to an input interface variable.
         In addition to updating the reflection data, this function modifies
         the underlying SPIR-V bytecode. The updated code can be retrieved
         with spvReflectGetCode().
         It is the caller's responsibility to avoid assigning the same
         location to multiple input variables.  If the input variable is used
         by multiple entry points in the module, it will be changed in all of
         them.
 @param  p_module          Pointer to an instance of SpvReflectShaderModule.
 @param  p_input_variable  Pointer to the input variable to update.
 @param  new_location      The new location to assign to p_input_variable.
 @return                   If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                           Otherwise, the error code indicates the cause of
                           the failure.

*/
SpvReflectResult spvReflectChangeInputVariableLocation(
  SpvReflectShaderModule*            p_module,
  const SpvReflectInterfaceVariable* p_input_variable,
  uint32_t                           new_location
);


/*! @fn spvReflectChangeOutputVariableLocation
 @brief  Assign a new location to an output interface variable.
         In addition to updating the reflection data, this function modifies
         the underlying SPIR-V bytecode. The updated code can be retrieved
         with spvReflectGetCode().
         It is the caller's responsibility to avoid assigning the same
         location to multiple output variables.  If the output variable is used
         by multiple entry points in the module, it will be changed in all of
         them.
 @param  p_module          Pointer to an instance of SpvReflectShaderModule.
 @param  p_output_variable Pointer to the output variable to update.
 @param  new_location      The new location to assign to p_output_variable.
 @return                   If successful, returns SPV_REFLECT_RESULT_SUCCESS.
                           Otherwise, the error code indicates the cause of
                           the failure.

*/
SpvReflectResult spvReflectChangeOutputVariableLocation(
  SpvReflectShaderModule*             p_module,
  const SpvReflectInterfaceVariable*  p_output_variable,
  uint32_t                            new_location
);


/*! @fn spvReflectSourceLanguage

 @param  source_lang  The source language code.
 @return Returns string of source language specified in \a source_lang.
         The caller must not free the memory associated with this string.
*/
const char* spvReflectSourceLanguage(SpvSourceLanguage source_lang);

/*! @fn spvReflectBlockVariableTypeName

 @param  p_var Pointer to block variable.
 @return Returns string of block variable's type description type name
         or NULL if p_var is NULL.
*/
const char* spvReflectBlockVariableTypeName(
  const SpvReflectBlockVariable* p_var
);

#if defined(__cplusplus)
};
#endif

#if defined(__cplusplus)
#include <cstdlib>
#include <string>
#include <vector>

namespace spv_reflect {

/*! \class ShaderModule

*/
class ShaderModule {
public:
  ShaderModule();
  ShaderModule(size_t size, const void* p_code, SpvReflectModuleFlags flags = SPV_REFLECT_MODULE_FLAG_NONE);
  ShaderModule(const std::vector<uint8_t>& code, SpvReflectModuleFlags flags = SPV_REFLECT_MODULE_FLAG_NONE);
  ShaderModule(const std::vector<uint32_t>& code, SpvReflectModuleFlags flags = SPV_REFLECT_MODULE_FLAG_NONE);
  ~ShaderModule();

  ShaderModule(ShaderModule&& other);
  ShaderModule& operator=(ShaderModule&& other);

  SpvReflectResult GetResult() const;

  const SpvReflectShaderModule& GetShaderModule() const;

  uint32_t        GetCodeSize() const;
  const uint32_t* GetCode() const;

  const char*           GetEntryPointName() const;

  const char*           GetSourceFile() const;

  uint32_t                      GetEntryPointCount() const;
  const char*                   GetEntryPointName(uint32_t index) const;
  SpvReflectShaderStageFlagBits GetEntryPointShaderStage(uint32_t index) const;

  SpvReflectShaderStageFlagBits GetShaderStage() const;
  SPV_REFLECT_DEPRECATED("Renamed to GetShaderStage")
  SpvReflectShaderStageFlagBits GetVulkanShaderStage() const {
    return GetShaderStage();
  }

  SpvReflectResult  EnumerateDescriptorBindings(uint32_t* p_count, SpvReflectDescriptorBinding** pp_bindings) const;
  SpvReflectResult  EnumerateEntryPointDescriptorBindings(const char* entry_point, uint32_t* p_count, SpvReflectDescriptorBinding** pp_bindings) const;
  SpvReflectResult  EnumerateDescriptorSets( uint32_t* p_count, SpvReflectDescriptorSet** pp_sets) const ;
  SpvReflectResult  EnumerateEntryPointDescriptorSets(const char* entry_point, uint32_t* p_count, SpvReflectDescriptorSet** pp_sets) const ;
  SpvReflectResult  EnumerateInterfaceVariables(uint32_t* p_count, SpvReflectInterfaceVariable** pp_variables) const;
  SpvReflectResult  EnumerateEntryPointInterfaceVariables(const char* entry_point, uint32_t* p_count, SpvReflectInterfaceVariable** pp_variables) const;
  SpvReflectResult  EnumerateInputVariables(uint32_t* p_count,SpvReflectInterfaceVariable** pp_variables) const;
  SpvReflectResult  EnumerateEntryPointInputVariables(const char* entry_point, uint32_t* p_count, SpvReflectInterfaceVariable** pp_variables) const;
  SpvReflectResult  EnumerateOutputVariables(uint32_t* p_count,SpvReflectInterfaceVariable** pp_variables) const;
  SpvReflectResult  EnumerateEntryPointOutputVariables(const char* entry_point, uint32_t* p_count, SpvReflectInterfaceVariable** pp_variables) const;
  SpvReflectResult  EnumeratePushConstantBlocks(uint32_t* p_count, SpvReflectBlockVariable** pp_blocks) const;
  SpvReflectResult  EnumerateEntryPointPushConstantBlocks(const char* entry_point, uint32_t* p_count, SpvReflectBlockVariable** pp_blocks) const;
  SPV_REFLECT_DEPRECATED("Renamed to EnumeratePushConstantBlocks")
  SpvReflectResult  EnumeratePushConstants(uint32_t* p_count, SpvReflectBlockVariable** pp_blocks) const {
    return EnumeratePushConstantBlocks(p_count, pp_blocks);
  }

  const SpvReflectDescriptorBinding*  GetDescriptorBinding(uint32_t binding_number, uint32_t set_number, SpvReflectResult* p_result = nullptr) const;
  const SpvReflectDescriptorBinding*  GetEntryPointDescriptorBinding(const char* entry_point, uint32_t binding_number, uint32_t set_number, SpvReflectResult* p_result = nullptr) const;
  const SpvReflectDescriptorSet*      GetDescriptorSet(uint32_t set_number, SpvReflectResult* p_result = nullptr) const;
  const SpvReflectDescriptorSet*      GetEntryPointDescriptorSet(const char* entry_point, uint32_t set_number, SpvReflectResult* p_result = nullptr) const;
  const SpvReflectInterfaceVariable*  GetInputVariableByLocation(uint32_t location,  SpvReflectResult* p_result = nullptr) const;
  SPV_REFLECT_DEPRECATED("Renamed to GetInputVariableByLocation")
  const SpvReflectInterfaceVariable*  GetInputVariable(uint32_t location,  SpvReflectResult* p_result = nullptr) const {
    return GetInputVariableByLocation(location, p_result);
  }
  const SpvReflectInterfaceVariable*  GetEntryPointInputVariableByLocation(const char* entry_point, uint32_t location,  SpvReflectResult* p_result = nullptr) const;
  const SpvReflectInterfaceVariable*  GetInputVariableBySemantic(const char* semantic,  SpvReflectResult* p_result = nullptr) const;
  const SpvReflectInterfaceVariable*  GetEntryPointInputVariableBySemantic(const char* entry_point, const char* semantic,  SpvReflectResult* p_result = nullptr) const;
  const SpvReflectInterfaceVariable*  GetOutputVariableByLocation(uint32_t location, SpvReflectResult*  p_result = nullptr) const;
  SPV_REFLECT_DEPRECATED("Renamed to GetOutputVariableByLocation")
  const SpvReflectInterfaceVariable*  GetOutputVariable(uint32_t location, SpvReflectResult*  p_result = nullptr) const {
    return GetOutputVariableByLocation(location, p_result);
  }
  const SpvReflectInterfaceVariable*  GetEntryPointOutputVariableByLocation(const char* entry_point, uint32_t location, SpvReflectResult*  p_result = nullptr) const;
  const SpvReflectInterfaceVariable*  GetOutputVariableBySemantic(const char* semantic, SpvReflectResult*  p_result = nullptr) const;
  const SpvReflectInterfaceVariable*  GetEntryPointOutputVariableBySemantic(const char* entry_point, const char* semantic, SpvReflectResult*  p_result = nullptr) const;
  const SpvReflectBlockVariable*      GetPushConstantBlock(uint32_t index, SpvReflectResult*  p_result = nullptr) const;
  SPV_REFLECT_DEPRECATED("Renamed to GetPushConstantBlock")
  const SpvReflectBlockVariable*      GetPushConstant(uint32_t index, SpvReflectResult*  p_result = nullptr) const {
    return GetPushConstantBlock(index, p_result);
  }
  const SpvReflectBlockVariable*      GetEntryPointPushConstantBlock(const char* entry_point, SpvReflectResult*  p_result = nullptr) const;

  SpvReflectResult ChangeDescriptorBindingNumbers(const SpvReflectDescriptorBinding* p_binding,
      uint32_t new_binding_number = SPV_REFLECT_BINDING_NUMBER_DONT_CHANGE,
      uint32_t optional_new_set_number = SPV_REFLECT_SET_NUMBER_DONT_CHANGE);
  SPV_REFLECT_DEPRECATED("Renamed to ChangeDescriptorBindingNumbers")
  SpvReflectResult ChangeDescriptorBindingNumber(const SpvReflectDescriptorBinding* p_binding, uint32_t new_binding_number = SPV_REFLECT_BINDING_NUMBER_DONT_CHANGE,
      uint32_t new_set_number = SPV_REFLECT_SET_NUMBER_DONT_CHANGE) {
    return ChangeDescriptorBindingNumbers(p_binding, new_binding_number, new_set_number);
  }
  SpvReflectResult ChangeDescriptorSetNumber(const SpvReflectDescriptorSet* p_set, uint32_t new_set_number = SPV_REFLECT_SET_NUMBER_DONT_CHANGE);
  SpvReflectResult ChangeInputVariableLocation(const SpvReflectInterfaceVariable* p_input_variable, uint32_t new_location);
  SpvReflectResult ChangeOutputVariableLocation(const SpvReflectInterfaceVariable* p_output_variable, uint32_t new_location);

private:
  // Make noncopyable
  ShaderModule(const ShaderModule&);
  ShaderModule& operator=(const ShaderModule&);

private:
  mutable SpvReflectResult  m_result = SPV_REFLECT_RESULT_NOT_READY;
  SpvReflectShaderModule    m_module = {};
};


// =================================================================================================
// ShaderModule
// =================================================================================================

/*! @fn ShaderModule

*/
inline ShaderModule::ShaderModule() {}


/*! @fn ShaderModule

  @param  size
  @param  p_code

*/
inline ShaderModule::ShaderModule(size_t size, const void* p_code, SpvReflectModuleFlags flags) {
  m_result = spvReflectCreateShaderModule2(
    flags,
    size,
    p_code,
    &m_module);
}

/*! @fn ShaderModule

  @param  code
  
*/
inline ShaderModule::ShaderModule(const std::vector<uint8_t>& code, SpvReflectModuleFlags flags) {
  m_result = spvReflectCreateShaderModule2(
    flags,
    code.size(),
    code.data(),
    &m_module);
}

/*! @fn ShaderModule

  @param  code
  
*/
inline ShaderModule::ShaderModule(const std::vector<uint32_t>& code, SpvReflectModuleFlags flags) {
  m_result = spvReflectCreateShaderModule2(
    flags,
    code.size() * sizeof(uint32_t),
    code.data(),
    &m_module);
}

/*! @fn  ~ShaderModule

*/
inline ShaderModule::~ShaderModule() {
  spvReflectDestroyShaderModule(&m_module);
}


inline ShaderModule::ShaderModule(ShaderModule&& other)
{
    *this = std::move(other);
}

inline ShaderModule& ShaderModule::operator=(ShaderModule&& other)
{
    m_result = std::move(other.m_result);
    m_module = std::move(other.m_module);

    other.m_module = {};
    return *this;
}

/*! @fn GetResult

  @return

*/
inline SpvReflectResult ShaderModule::GetResult() const {
  return m_result;
}


/*! @fn GetShaderModule

  @return

*/
inline const SpvReflectShaderModule& ShaderModule::GetShaderModule() const {
  return m_module;
}


/*! @fn GetCodeSize

  @return

  */
inline uint32_t ShaderModule::GetCodeSize() const {
  return spvReflectGetCodeSize(&m_module);
}


/*! @fn GetCode

  @return

*/
inline const uint32_t* ShaderModule::GetCode() const {
  return spvReflectGetCode(&m_module);
}


/*! @fn GetEntryPoint

  @return Returns entry point

*/
inline const char* ShaderModule::GetEntryPointName() const {
  return this->GetEntryPointName(0);
}

/*! @fn GetEntryPoint

  @return Returns entry point

*/
inline const char* ShaderModule::GetSourceFile() const {
  return m_module.source_file;
}

/*! @fn GetEntryPointCount

  @param
  @return
*/
inline uint32_t ShaderModule::GetEntryPointCount() const {
  return m_module.entry_point_count;
}

/*! @fn GetEntryPointName

  @param index
  @return
*/
inline const char* ShaderModule::GetEntryPointName(uint32_t index) const {
  return m_module.entry_points[index].name;
}

/*! @fn GetEntryPointShaderStage

  @param index
  @return Returns the shader stage for the entry point at \b index
*/
inline SpvReflectShaderStageFlagBits ShaderModule::GetEntryPointShaderStage(uint32_t index) const {
  return m_module.entry_points[index].shader_stage;
}

/*! @fn GetShaderStage

  @return Returns shader stage for the first entry point

*/
inline SpvReflectShaderStageFlagBits ShaderModule::GetShaderStage() const {
  return m_module.shader_stage;
}

/*! @fn EnumerateDescriptorBindings

  @param  count
  @param  p_binding_numbers
  @param  pp_bindings
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateDescriptorBindings(
  uint32_t*                     p_count,
  SpvReflectDescriptorBinding** pp_bindings
) const
{
  m_result = spvReflectEnumerateDescriptorBindings(
    &m_module,
    p_count,
    pp_bindings);
  return m_result;
}

/*! @fn EnumerateEntryPointDescriptorBindings

  @param  entry_point
  @param  count
  @param  pp_bindings
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateEntryPointDescriptorBindings(
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectDescriptorBinding** pp_bindings
) const
{
  m_result = spvReflectEnumerateEntryPointDescriptorBindings(
      &m_module,
      entry_point,
      p_count,
      pp_bindings);
  return m_result;
}


/*! @fn EnumerateDescriptorSets

  @param  count
  @param  pp_sets
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateDescriptorSets(
  uint32_t*                 p_count,
  SpvReflectDescriptorSet** pp_sets
) const
{
  m_result = spvReflectEnumerateDescriptorSets(
    &m_module,
    p_count,
    pp_sets);
  return m_result;
}

/*! @fn EnumerateEntryPointDescriptorSets

  @param  entry_point
  @param  count
  @param  pp_sets
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateEntryPointDescriptorSets(
  const char*               entry_point,
  uint32_t*                 p_count,
  SpvReflectDescriptorSet** pp_sets
) const
{
  m_result = spvReflectEnumerateEntryPointDescriptorSets(
      &m_module,
      entry_point,
      p_count,
      pp_sets);
  return m_result;
}


/*! @fn EnumerateInterfaceVariables

  @param  count
  @param  pp_variables
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateInterfaceVariables(
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
) const
{
  m_result = spvReflectEnumerateInterfaceVariables(
    &m_module,
    p_count,
    pp_variables);
  return m_result;
}

/*! @fn EnumerateEntryPointInterfaceVariables

  @param  entry_point
  @param  count
  @param  pp_variables
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateEntryPointInterfaceVariables(
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
) const
{
  m_result = spvReflectEnumerateEntryPointInterfaceVariables(
      &m_module,
      entry_point,
      p_count,
      pp_variables);
  return m_result;
}


/*! @fn EnumerateInputVariables

  @param  count
  @param  pp_variables
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateInputVariables(
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
) const
{
  m_result = spvReflectEnumerateInputVariables(
    &m_module,
    p_count,
    pp_variables);
  return m_result;
}

/*! @fn EnumerateEntryPointInputVariables

  @param  entry_point
  @param  count
  @param  pp_variables
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateEntryPointInputVariables(
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
) const
{
  m_result = spvReflectEnumerateEntryPointInputVariables(
      &m_module,
      entry_point,
      p_count,
      pp_variables);
  return m_result;
}


/*! @fn EnumerateOutputVariables

  @param  count
  @param  pp_variables
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateOutputVariables(
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
) const
{
  m_result = spvReflectEnumerateOutputVariables(
    &m_module,
    p_count,
    pp_variables);
  return m_result;
}

/*! @fn EnumerateEntryPointOutputVariables

  @param  entry_point
  @param  count
  @param  pp_variables
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateEntryPointOutputVariables(
  const char*                   entry_point,
  uint32_t*                     p_count,
  SpvReflectInterfaceVariable** pp_variables
) const
{
  m_result = spvReflectEnumerateEntryPointOutputVariables(
      &m_module,
      entry_point,
      p_count,
      pp_variables);
  return m_result;
}


/*! @fn EnumeratePushConstantBlocks

  @param  count
  @param  pp_blocks
  @return

*/
inline SpvReflectResult ShaderModule::EnumeratePushConstantBlocks(
  uint32_t*                 p_count,
  SpvReflectBlockVariable** pp_blocks
) const
{
  m_result = spvReflectEnumeratePushConstantBlocks(
    &m_module,
    p_count,
    pp_blocks);
  return m_result;
}

/*! @fn EnumerateEntryPointPushConstantBlocks

  @param  entry_point
  @param  count
  @param  pp_blocks
  @return

*/
inline SpvReflectResult ShaderModule::EnumerateEntryPointPushConstantBlocks(
  const char*               entry_point,
  uint32_t*                 p_count,
  SpvReflectBlockVariable** pp_blocks
) const
{
  m_result = spvReflectEnumerateEntryPointPushConstantBlocks(
      &m_module,
      entry_point,
      p_count,
      pp_blocks);
  return m_result;
}


/*! @fn GetDescriptorBinding

  @param  binding_number
  @param  set_number
  @param  p_result
  @return

*/
inline const SpvReflectDescriptorBinding* ShaderModule::GetDescriptorBinding(
  uint32_t          binding_number,
  uint32_t          set_number,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetDescriptorBinding(
    &m_module,
    binding_number,
    set_number,
    p_result);
}

/*! @fn GetEntryPointDescriptorBinding

  @param  entry_point
  @param  binding_number
  @param  set_number
  @param  p_result
  @return

*/
inline const SpvReflectDescriptorBinding* ShaderModule::GetEntryPointDescriptorBinding(
  const char*       entry_point,
  uint32_t          binding_number,
  uint32_t          set_number,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetEntryPointDescriptorBinding(
    &m_module,
    entry_point,
    binding_number,
    set_number,
    p_result);
}


/*! @fn GetDescriptorSet

  @param  set_number
  @param  p_result
  @return

*/
inline const SpvReflectDescriptorSet* ShaderModule::GetDescriptorSet(
  uint32_t          set_number,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetDescriptorSet(
    &m_module,
    set_number,
    p_result);
}

/*! @fn GetEntryPointDescriptorSet

  @param  entry_point
  @param  set_number
  @param  p_result
  @return

*/
inline const SpvReflectDescriptorSet* ShaderModule::GetEntryPointDescriptorSet(
  const char*       entry_point,
  uint32_t          set_number,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetEntryPointDescriptorSet(
    &m_module,
    entry_point,
    set_number,
    p_result);
}


/*! @fn GetInputVariable

  @param  location
  @param  p_result
  @return

*/
inline const SpvReflectInterfaceVariable* ShaderModule::GetInputVariableByLocation(
  uint32_t          location,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetInputVariableByLocation(
    &m_module,
    location,
    p_result);
}
inline const SpvReflectInterfaceVariable* ShaderModule::GetInputVariableBySemantic(
  const char*       semantic,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetInputVariableBySemantic(
    &m_module,
    semantic,
    p_result);
}

/*! @fn GetEntryPointInputVariable

  @param  entry_point
  @param  location
  @param  p_result
  @return

*/
inline const SpvReflectInterfaceVariable* ShaderModule::GetEntryPointInputVariableByLocation(
  const char*       entry_point,
  uint32_t          location,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetEntryPointInputVariableByLocation(
    &m_module,
    entry_point,
    location,
    p_result);
}
inline const SpvReflectInterfaceVariable* ShaderModule::GetEntryPointInputVariableBySemantic(
  const char*       entry_point,
  const char*       semantic,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetEntryPointInputVariableBySemantic(
    &m_module,
    entry_point,
    semantic,
    p_result);
}


/*! @fn GetOutputVariable

  @param  location
  @param  p_result
  @return

*/
inline const SpvReflectInterfaceVariable* ShaderModule::GetOutputVariableByLocation(
  uint32_t           location,
  SpvReflectResult*  p_result
) const
{
  return spvReflectGetOutputVariableByLocation(
    &m_module,
    location,
    p_result);
}
inline const SpvReflectInterfaceVariable* ShaderModule::GetOutputVariableBySemantic(
  const char*       semantic,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetOutputVariableBySemantic(&m_module,
    semantic,
    p_result);
}

/*! @fn GetEntryPointOutputVariable

  @param  entry_point
  @param  location
  @param  p_result
  @return

*/
inline const SpvReflectInterfaceVariable* ShaderModule::GetEntryPointOutputVariableByLocation(
  const char*        entry_point,
  uint32_t           location,
  SpvReflectResult*  p_result
) const
{
  return spvReflectGetEntryPointOutputVariableByLocation(
    &m_module,
    entry_point,
    location,
    p_result);
}
inline const SpvReflectInterfaceVariable* ShaderModule::GetEntryPointOutputVariableBySemantic(
  const char*       entry_point,
  const char*       semantic,
  SpvReflectResult* p_result
) const
{
  return spvReflectGetEntryPointOutputVariableBySemantic(
    &m_module,
    entry_point,
    semantic,
    p_result);
}


/*! @fn GetPushConstant

  @param  index
  @param  p_result
  @return

*/
inline const SpvReflectBlockVariable* ShaderModule::GetPushConstantBlock(
  uint32_t           index,
  SpvReflectResult*  p_result
) const
{
  return spvReflectGetPushConstantBlock(
    &m_module,
    index,
    p_result);
}

/*! @fn GetEntryPointPushConstant

  @param  entry_point
  @param  index
  @param  p_result
  @return

*/
inline const SpvReflectBlockVariable* ShaderModule::GetEntryPointPushConstantBlock(
  const char*        entry_point,
  SpvReflectResult*  p_result
) const
{
  return spvReflectGetEntryPointPushConstantBlock(
    &m_module,
    entry_point,
    p_result);
}


/*! @fn ChangeDescriptorBindingNumbers

  @param  p_binding
  @param  new_binding_number
  @param  new_set_number
  @return

*/
inline SpvReflectResult ShaderModule::ChangeDescriptorBindingNumbers(
  const SpvReflectDescriptorBinding* p_binding,
  uint32_t                           new_binding_number,
  uint32_t                           new_set_number
)
{
  return spvReflectChangeDescriptorBindingNumbers(
    &m_module,
    p_binding,
    new_binding_number,
    new_set_number);
}


/*! @fn ChangeDescriptorSetNumber

  @param  p_set
  @param  new_set_number
  @return

*/
inline SpvReflectResult ShaderModule::ChangeDescriptorSetNumber(
  const SpvReflectDescriptorSet* p_set,
  uint32_t                       new_set_number
)
{
  return spvReflectChangeDescriptorSetNumber(
    &m_module,
    p_set,
    new_set_number);
}


/*! @fn ChangeInputVariableLocation

  @param  p_input_variable
  @param  new_location
  @return

*/
inline SpvReflectResult ShaderModule::ChangeInputVariableLocation(
  const SpvReflectInterfaceVariable* p_input_variable,
  uint32_t                           new_location)
{
  return spvReflectChangeInputVariableLocation(
    &m_module,
    p_input_variable,
    new_location);
}


/*! @fn ChangeOutputVariableLocation

  @param  p_input_variable
  @param  new_location
  @return

*/
inline SpvReflectResult ShaderModule::ChangeOutputVariableLocation(
  const SpvReflectInterfaceVariable* p_output_variable,
  uint32_t                           new_location)
{
  return spvReflectChangeOutputVariableLocation(
    &m_module,
    p_output_variable,
    new_location);
}

} // namespace spv_reflect
#endif // defined(__cplusplus)
#endif // SPIRV_REFLECT_H
