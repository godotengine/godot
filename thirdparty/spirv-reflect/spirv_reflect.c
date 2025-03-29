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

#include "spirv_reflect.h"

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#if defined(WIN32)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include <stdlib.h>
#else
#include <stdlib.h>
#endif

#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 7) || defined(__APPLE_CC__)
#define FALLTHROUGH __attribute__((fallthrough))
#else
#define FALLTHROUGH
#endif

#if defined(SPIRV_REFLECT_ENABLE_ASSERTS)
#define SPV_REFLECT_ASSERT(COND) assert(COND);
#else
#define SPV_REFLECT_ASSERT(COND)
#endif

// clang-format off
enum {
  SPIRV_STARTING_WORD_INDEX       = 5,
  SPIRV_WORD_SIZE                 = sizeof(uint32_t),
  SPIRV_BYTE_WIDTH                = 8,
  SPIRV_MINIMUM_FILE_SIZE         = SPIRV_STARTING_WORD_INDEX * SPIRV_WORD_SIZE,
  SPIRV_DATA_ALIGNMENT            = 4 * SPIRV_WORD_SIZE, // 16
  SPIRV_ACCESS_CHAIN_INDEX_OFFSET = 4,
};

enum {
  INVALID_VALUE  = 0xFFFFFFFF,
};

enum {
  MAX_NODE_NAME_LENGTH        = 1024,
  // Number of unique PhysicalStorageBuffer structs tracked to detect recursion
  MAX_RECURSIVE_PHYSICAL_POINTER_CHECK = 128,
};

enum {
  IMAGE_SAMPLED = 1,
  IMAGE_STORAGE = 2,
};

typedef struct SpvReflectPrvArrayTraits {
  uint32_t                        element_type_id;
  uint32_t                        length_id;
} SpvReflectPrvArrayTraits;

typedef struct SpvReflectPrvImageTraits {
  uint32_t                        sampled_type_id;
  SpvDim                          dim;
  uint32_t                        depth;
  uint32_t                        arrayed;
  uint32_t                        ms;
  uint32_t                        sampled;
  SpvImageFormat                  image_format;
} SpvReflectPrvImageTraits;

typedef struct SpvReflectPrvNumberDecoration {
  uint32_t                        word_offset;
  uint32_t                        value;
} SpvReflectPrvNumberDecoration;

typedef struct SpvReflectPrvStringDecoration {
  uint32_t                        word_offset;
  const char*                     value;
} SpvReflectPrvStringDecoration;

typedef struct SpvReflectPrvDecorations {
  bool                            is_relaxed_precision;
  bool                            is_block;
  bool                            is_buffer_block;
  bool                            is_row_major;
  bool                            is_column_major;
  bool                            is_built_in;
  bool                            is_noperspective;
  bool                            is_flat;
  bool                            is_non_writable;
  bool                            is_non_readable;
  bool                            is_patch;
  bool                            is_per_vertex;
  bool                            is_per_task;
  bool                            is_weight_texture;
  bool                            is_block_match_texture;
  SpvReflectUserType              user_type;
  SpvReflectPrvNumberDecoration   set;
  SpvReflectPrvNumberDecoration   binding;
  SpvReflectPrvNumberDecoration   input_attachment_index;
  SpvReflectPrvNumberDecoration   location;
  SpvReflectPrvNumberDecoration   component;
  SpvReflectPrvNumberDecoration   offset;
  SpvReflectPrvNumberDecoration   uav_counter_buffer;
  SpvReflectPrvStringDecoration   semantic;
  uint32_t                        array_stride;
  uint32_t                        matrix_stride;
  uint32_t                        spec_id;
  SpvBuiltIn                      built_in;
} SpvReflectPrvDecorations;

typedef struct SpvReflectPrvNode {
  uint32_t                        result_id;
  SpvOp                           op;
  uint32_t                        result_type_id;
  uint32_t                        type_id;
  SpvCapability                   capability;
  SpvStorageClass                 storage_class;
  uint32_t                        word_offset;
  uint32_t                        word_count;
  bool                            is_type;

  SpvReflectPrvArrayTraits        array_traits;
  SpvReflectPrvImageTraits        image_traits;
  uint32_t                        image_type_id;

  const char*                     name;
  SpvReflectPrvDecorations        decorations;
  uint32_t                        member_count;
  const char**                    member_names;
  SpvReflectPrvDecorations*       member_decorations;
} SpvReflectPrvNode;

typedef struct SpvReflectPrvString {
  uint32_t                        result_id;
  const char*                     string;
} SpvReflectPrvString;

// There are a limit set of instructions that can touch an OpVariable,
// these are represented here with how it was accessed
// Examples:
//    OpImageRead  -> OpLoad -> OpVariable
//    OpImageWrite -> OpLoad -> OpVariable
//    OpStore      -> OpAccessChain -> OpAccessChain -> OpVariable
//    OpAtomicIAdd -> OpAccessChain -> OpVariable
//    OpAtomicLoad -> OpImageTexelPointer -> OpVariable
typedef struct SpvReflectPrvAccessedVariable {
  SpvReflectPrvNode*     p_node;
  uint32_t               result_id;
  uint32_t               variable_ptr;
} SpvReflectPrvAccessedVariable;

typedef struct SpvReflectPrvFunction {
  uint32_t                        id;
  uint32_t                        callee_count;
  uint32_t*                       callees;
  struct SpvReflectPrvFunction**  callee_ptrs;
  uint32_t                        accessed_variable_count;
  SpvReflectPrvAccessedVariable*  accessed_variables;
} SpvReflectPrvFunction;

typedef struct SpvReflectPrvAccessChain {
  uint32_t                        result_id;
  uint32_t                        result_type_id;
  //
  // Pointing to the base of a composite object.
  // Generally the id of descriptor block variable
  uint32_t                        base_id;
  //
  // From spec:
  //   The first index in Indexes will select the
  //   top-level member/element/component/element
  //   of the base composite
  uint32_t                        index_count;
  uint32_t*                       indexes;
  //
  // Block variable ac is pointing to (for block references)
  SpvReflectBlockVariable*        block_var;
} SpvReflectPrvAccessChain;

// To prevent infinite recursion, we never walk down a
// PhysicalStorageBuffer struct twice, but incase a 2nd variable
// needs to use that struct, save a copy
typedef struct SpvReflectPrvPhysicalPointerStruct {
    uint32_t struct_id;
    // first variable to see the PhysicalStorageBuffer struct
    SpvReflectBlockVariable* p_var;
} SpvReflectPrvPhysicalPointerStruct;

typedef struct SpvReflectPrvParser {
  size_t                          spirv_word_count;
  uint32_t*                       spirv_code;
  uint32_t                        string_count;
  SpvReflectPrvString*            strings;
  SpvSourceLanguage               source_language;
  uint32_t                        source_language_version;
  uint32_t                        source_file_id;
  const char*                     source_embedded;
  size_t                          node_count;
  SpvReflectPrvNode*              nodes;
  uint32_t                        entry_point_count;
  uint32_t                        capability_count;
  uint32_t                        function_count;
  SpvReflectPrvFunction*          functions;
  uint32_t                        access_chain_count;
  SpvReflectPrvAccessChain*       access_chains;

  uint32_t                        type_count;
  uint32_t                        descriptor_count;
  uint32_t                        push_constant_count;

  SpvReflectTypeDescription*      physical_pointer_check[MAX_RECURSIVE_PHYSICAL_POINTER_CHECK];
  uint32_t                        physical_pointer_count;

  SpvReflectPrvPhysicalPointerStruct* physical_pointer_structs;
  uint32_t                            physical_pointer_struct_count;
} SpvReflectPrvParser;
// clang-format on

static uint32_t Max(uint32_t a, uint32_t b) { return a > b ? a : b; }
static uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

static uint32_t RoundUp(uint32_t value, uint32_t multiple) {
  assert(multiple && ((multiple & (multiple - 1)) == 0));
  return (value + multiple - 1) & ~(multiple - 1);
}

#define IsNull(ptr) (ptr == NULL)

#define IsNotNull(ptr) (ptr != NULL)

#define SafeFree(ptr) \
  {                   \
    free((void*)ptr); \
    ptr = NULL;       \
  }

static int SortCompareUint32(const void* a, const void* b) {
  const uint32_t* p_a = (const uint32_t*)a;
  const uint32_t* p_b = (const uint32_t*)b;

  return (int)*p_a - (int)*p_b;
}

static int SortCompareAccessedVariable(const void* a, const void* b) {
  const SpvReflectPrvAccessedVariable* p_a = (const SpvReflectPrvAccessedVariable*)a;
  const SpvReflectPrvAccessedVariable* p_b = (const SpvReflectPrvAccessedVariable*)b;

  return (int)p_a->variable_ptr - (int)p_b->variable_ptr;
}

//
// De-duplicates a sorted array and returns the new size.
//
// Note: The array doesn't actually need to be sorted, just
// arranged into "runs" so that all the entries with one
// value are adjacent.
//
static size_t DedupSortedUint32(uint32_t* arr, size_t size) {
  if (size == 0) {
    return 0;
  }
  size_t dedup_idx = 0;
  for (size_t i = 0; i < size; ++i) {
    if (arr[dedup_idx] != arr[i]) {
      ++dedup_idx;
      arr[dedup_idx] = arr[i];
    }
  }
  return dedup_idx + 1;
}

static bool SearchSortedUint32(const uint32_t* arr, size_t size, uint32_t target) {
  size_t lo = 0;
  size_t hi = size;
  while (lo < hi) {
    size_t mid = (hi - lo) / 2 + lo;
    if (arr[mid] == target) {
      return true;
    } else if (arr[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return false;
}

static SpvReflectResult IntersectSortedAccessedVariable(const SpvReflectPrvAccessedVariable* p_arr0, size_t arr0_size,
                                                        const uint32_t* p_arr1, size_t arr1_size, uint32_t** pp_res,
                                                        size_t* res_size) {
  *pp_res = NULL;
  *res_size = 0;
  if (IsNull(p_arr0) || IsNull(p_arr1)) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  const SpvReflectPrvAccessedVariable* p_arr0_end = p_arr0 + arr0_size;
  const uint32_t* p_arr1_end = p_arr1 + arr1_size;

  const SpvReflectPrvAccessedVariable* p_idx0 = p_arr0;
  const uint32_t* p_idx1 = p_arr1;
  while (p_idx0 != p_arr0_end && p_idx1 != p_arr1_end) {
    if (p_idx0->variable_ptr < *p_idx1) {
      ++p_idx0;
    } else if (p_idx0->variable_ptr > *p_idx1) {
      ++p_idx1;
    } else {
      ++*res_size;
      ++p_idx0;
      ++p_idx1;
    }
  }

  if (*res_size > 0) {
    *pp_res = (uint32_t*)calloc(*res_size, sizeof(**pp_res));
    if (IsNull(*pp_res)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
    uint32_t* p_idxr = *pp_res;
    p_idx0 = p_arr0;
    p_idx1 = p_arr1;
    while (p_idx0 != p_arr0_end && p_idx1 != p_arr1_end) {
      if (p_idx0->variable_ptr < *p_idx1) {
        ++p_idx0;
      } else if (p_idx0->variable_ptr > *p_idx1) {
        ++p_idx1;
      } else {
        *(p_idxr++) = p_idx0->variable_ptr;
        ++p_idx0;
        ++p_idx1;
      }
    }
  }
  return SPV_REFLECT_RESULT_SUCCESS;
}

static bool InRange(const SpvReflectPrvParser* p_parser, uint32_t index) {
  bool in_range = false;
  if (IsNotNull(p_parser)) {
    in_range = (index < p_parser->spirv_word_count);
  }
  return in_range;
}

static SpvReflectResult ReadU32(SpvReflectPrvParser* p_parser, uint32_t word_offset, uint32_t* p_value) {
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->spirv_code));
  assert(InRange(p_parser, word_offset));
  SpvReflectResult result = SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_EOF;
  if (IsNotNull(p_parser) && IsNotNull(p_parser->spirv_code) && InRange(p_parser, word_offset)) {
    *p_value = *(p_parser->spirv_code + word_offset);
    result = SPV_REFLECT_RESULT_SUCCESS;
  }
  return result;
}

#define UNCHECKED_READU32(parser, word_offset, value) \
  { (void)ReadU32(parser, word_offset, (uint32_t*)&(value)); }

#define CHECKED_READU32(parser, word_offset, value)                                              \
  {                                                                                              \
    SpvReflectResult checked_readu32_result = ReadU32(parser, word_offset, (uint32_t*)&(value)); \
    if (checked_readu32_result != SPV_REFLECT_RESULT_SUCCESS) {                                  \
      return checked_readu32_result;                                                             \
    }                                                                                            \
  }

#define CHECKED_READU32_CAST(parser, word_offset, cast_to_type, value)                                                   \
  {                                                                                                                      \
    uint32_t checked_readu32_cast_u32 = UINT32_MAX;                                                                      \
    SpvReflectResult checked_readu32_cast_result = ReadU32(parser, word_offset, (uint32_t*)&(checked_readu32_cast_u32)); \
    if (checked_readu32_cast_result != SPV_REFLECT_RESULT_SUCCESS) {                                                     \
      return checked_readu32_cast_result;                                                                                \
    }                                                                                                                    \
    value = (cast_to_type)checked_readu32_cast_u32;                                                                      \
  }

#define IF_READU32(result, parser, word_offset, value)          \
  if ((result) == SPV_REFLECT_RESULT_SUCCESS) {                 \
    result = ReadU32(parser, word_offset, (uint32_t*)&(value)); \
  }

#define IF_READU32_CAST(result, parser, word_offset, cast_to_type, value) \
  if ((result) == SPV_REFLECT_RESULT_SUCCESS) {                           \
    uint32_t if_readu32_cast_u32 = UINT32_MAX;                            \
    result = ReadU32(parser, word_offset, &if_readu32_cast_u32);          \
    if ((result) == SPV_REFLECT_RESULT_SUCCESS) {                         \
      value = (cast_to_type)if_readu32_cast_u32;                          \
    }                                                                     \
  }

static SpvReflectResult ReadStr(SpvReflectPrvParser* p_parser, uint32_t word_offset, uint32_t word_index, uint32_t word_count,
                                uint32_t* p_buf_size, char* p_buf) {
  uint32_t limit = (word_offset + word_count);
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->spirv_code));
  assert(InRange(p_parser, limit));
  SpvReflectResult result = SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_EOF;
  if (IsNotNull(p_parser) && IsNotNull(p_parser->spirv_code) && InRange(p_parser, limit)) {
    const char* c_str = (const char*)(p_parser->spirv_code + word_offset + word_index);
    uint32_t n = word_count * SPIRV_WORD_SIZE;
    uint32_t length_with_terminator = 0;
    for (uint32_t i = 0; i < n; ++i) {
      char c = *(c_str + i);
      if (c == 0) {
        length_with_terminator = i + 1;
        break;
      }
    }

    if (length_with_terminator > 0) {
      result = SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
      if (IsNotNull(p_buf_size) && IsNotNull(p_buf)) {
        result = SPV_REFLECT_RESULT_ERROR_RANGE_EXCEEDED;
        if (length_with_terminator <= *p_buf_size) {
          memset(p_buf, 0, *p_buf_size);
          memcpy(p_buf, c_str, length_with_terminator);
          result = SPV_REFLECT_RESULT_SUCCESS;
        }
      } else {
        if (IsNotNull(p_buf_size)) {
          *p_buf_size = length_with_terminator;
          result = SPV_REFLECT_RESULT_SUCCESS;
        }
      }
    }
  }
  return result;
}

static SpvReflectDecorationFlags ApplyDecorations(const SpvReflectPrvDecorations* p_decoration_fields) {
  SpvReflectDecorationFlags decorations = SPV_REFLECT_DECORATION_NONE;
  if (p_decoration_fields->is_relaxed_precision) {
    decorations |= SPV_REFLECT_DECORATION_RELAXED_PRECISION;
  }
  if (p_decoration_fields->is_block) {
    decorations |= SPV_REFLECT_DECORATION_BLOCK;
  }
  if (p_decoration_fields->is_buffer_block) {
    decorations |= SPV_REFLECT_DECORATION_BUFFER_BLOCK;
  }
  if (p_decoration_fields->is_row_major) {
    decorations |= SPV_REFLECT_DECORATION_ROW_MAJOR;
  }
  if (p_decoration_fields->is_column_major) {
    decorations |= SPV_REFLECT_DECORATION_COLUMN_MAJOR;
  }
  if (p_decoration_fields->is_built_in) {
    decorations |= SPV_REFLECT_DECORATION_BUILT_IN;
  }
  if (p_decoration_fields->is_noperspective) {
    decorations |= SPV_REFLECT_DECORATION_NOPERSPECTIVE;
  }
  if (p_decoration_fields->is_flat) {
    decorations |= SPV_REFLECT_DECORATION_FLAT;
  }
  if (p_decoration_fields->is_non_writable) {
    decorations |= SPV_REFLECT_DECORATION_NON_WRITABLE;
  }
  if (p_decoration_fields->is_non_readable) {
    decorations |= SPV_REFLECT_DECORATION_NON_READABLE;
  }
  if (p_decoration_fields->is_patch) {
    decorations |= SPV_REFLECT_DECORATION_PATCH;
  }
  if (p_decoration_fields->is_per_vertex) {
    decorations |= SPV_REFLECT_DECORATION_PER_VERTEX;
  }
  if (p_decoration_fields->is_per_task) {
    decorations |= SPV_REFLECT_DECORATION_PER_TASK;
  }
  if (p_decoration_fields->is_weight_texture) {
    decorations |= SPV_REFLECT_DECORATION_WEIGHT_TEXTURE;
  }
  if (p_decoration_fields->is_block_match_texture) {
    decorations |= SPV_REFLECT_DECORATION_BLOCK_MATCH_TEXTURE;
  }
  return decorations;
}

static void ApplyNumericTraits(const SpvReflectTypeDescription* p_type, SpvReflectNumericTraits* p_numeric_traits) {
  memcpy(p_numeric_traits, &p_type->traits.numeric, sizeof(p_type->traits.numeric));
}

static void ApplyArrayTraits(const SpvReflectTypeDescription* p_type, SpvReflectArrayTraits* p_array_traits) {
  memcpy(p_array_traits, &p_type->traits.array, sizeof(p_type->traits.array));
}

static bool IsSpecConstant(const SpvReflectPrvNode* p_node) {
  return (p_node->op == SpvOpSpecConstant || p_node->op == SpvOpSpecConstantOp || p_node->op == SpvOpSpecConstantTrue ||
          p_node->op == SpvOpSpecConstantFalse);
}

static SpvReflectPrvNode* FindNode(SpvReflectPrvParser* p_parser, uint32_t result_id) {
  SpvReflectPrvNode* p_node = NULL;
  for (size_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_elem = &(p_parser->nodes[i]);
    if (p_elem->result_id == result_id) {
      p_node = p_elem;
      break;
    }
  }
  return p_node;
}

static SpvReflectTypeDescription* FindType(SpvReflectShaderModule* p_module, uint32_t type_id) {
  SpvReflectTypeDescription* p_type = NULL;
  for (size_t i = 0; i < p_module->_internal->type_description_count; ++i) {
    SpvReflectTypeDescription* p_elem = &(p_module->_internal->type_descriptions[i]);
    if (p_elem->id == type_id) {
      p_type = p_elem;
      break;
    }
  }
  return p_type;
}

static SpvReflectPrvAccessChain* FindAccessChain(SpvReflectPrvParser* p_parser, uint32_t id) {
  const uint32_t ac_count = p_parser->access_chain_count;
  for (uint32_t i = 0; i < ac_count; i++) {
    if (p_parser->access_chains[i].result_id == id) {
      return &p_parser->access_chains[i];
    }
  }
  return 0;
}

// Access Chains mostly have their Base ID pointed directly to a OpVariable, but sometimes
// it will be through a load and this funciton handles the edge cases how to find that
static uint32_t FindAccessChainBaseVariable(SpvReflectPrvParser* p_parser, SpvReflectPrvAccessChain* p_access_chain) {
  uint32_t base_id = p_access_chain->base_id;
  SpvReflectPrvNode* base_node = FindNode(p_parser, base_id);
  // TODO - This is just a band-aid to fix crashes.
  // Need to understand why here and hopefully remove
  // https://github.com/KhronosGroup/SPIRV-Reflect/pull/206
  if (IsNull(base_node)) {
    return 0;
  }
  while (base_node->op != SpvOpVariable) {
    switch (base_node->op) {
      case SpvOpLoad: {
        UNCHECKED_READU32(p_parser, base_node->word_offset + 3, base_id);
      } break;
      case SpvOpFunctionParameter: {
        UNCHECKED_READU32(p_parser, base_node->word_offset + 2, base_id);
      } break;
      case SpvOpBitcast:
        // This can be caused by something like GL_EXT_buffer_reference_uvec2 trying to load a pointer.
        // We currently call from a push constant, so no way to have a reference loop back into the PC block
        return 0;
      default: {
        assert(false);
      } break;
    }

    SpvReflectPrvAccessChain* base_ac = FindAccessChain(p_parser, base_id);
    if (base_ac == 0) {
      return 0;
    }
    base_id = base_ac->base_id;
    base_node = FindNode(p_parser, base_id);
    if (IsNull(base_node)) {
      return 0;
    }
  }
  return base_id;
}

static SpvReflectBlockVariable* GetRefBlkVar(SpvReflectPrvParser* p_parser, SpvReflectPrvAccessChain* p_access_chain) {
  uint32_t base_id = p_access_chain->base_id;
  SpvReflectPrvNode* base_node = FindNode(p_parser, base_id);
  assert(base_node->op == SpvOpLoad);
  UNCHECKED_READU32(p_parser, base_node->word_offset + 3, base_id);
  SpvReflectPrvAccessChain* base_ac = FindAccessChain(p_parser, base_id);
  assert(base_ac != 0);
  SpvReflectBlockVariable* base_var = base_ac->block_var;
  assert(base_var != 0);
  return base_var;
}

bool IsPointerToPointer(SpvReflectPrvParser* p_parser, uint32_t type_id) {
  SpvReflectPrvNode* ptr_node = FindNode(p_parser, type_id);
  if (IsNull(ptr_node) || (ptr_node->op != SpvOpTypePointer)) {
    return false;
  }
  uint32_t pte_id = 0;
  UNCHECKED_READU32(p_parser, ptr_node->word_offset + 3, pte_id);
  SpvReflectPrvNode* pte_node = FindNode(p_parser, pte_id);
  if (IsNull(pte_node)) {
    return false;
  }
  return pte_node->op == SpvOpTypePointer;
}

static SpvReflectResult CreateParser(size_t size, void* p_code, SpvReflectPrvParser* p_parser) {
  if (p_code == NULL) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  if (size < SPIRV_MINIMUM_FILE_SIZE) {
    return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_CODE_SIZE;
  }
  if ((size % 4) != 0) {
    return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_CODE_SIZE;
  }

  p_parser->spirv_word_count = size / SPIRV_WORD_SIZE;
  p_parser->spirv_code = (uint32_t*)p_code;

  if (p_parser->spirv_code[0] != SpvMagicNumber) {
    return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_MAGIC_NUMBER;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static void DestroyParser(SpvReflectPrvParser* p_parser) {
  if (!IsNull(p_parser->nodes)) {
    // Free nodes
    for (size_t i = 0; i < p_parser->node_count; ++i) {
      SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
      if (IsNotNull(p_node->member_names)) {
        SafeFree(p_node->member_names);
      }
      if (IsNotNull(p_node->member_decorations)) {
        SafeFree(p_node->member_decorations);
      }
    }

    // Free functions
    for (size_t i = 0; i < p_parser->function_count; ++i) {
      SafeFree(p_parser->functions[i].callees);
      SafeFree(p_parser->functions[i].callee_ptrs);
      SafeFree(p_parser->functions[i].accessed_variables);
    }

    // Free access chains
    for (uint32_t i = 0; i < p_parser->access_chain_count; ++i) {
      SafeFree(p_parser->access_chains[i].indexes);
    }

    SafeFree(p_parser->nodes);
    SafeFree(p_parser->strings);
    SafeFree(p_parser->source_embedded);
    SafeFree(p_parser->functions);
    SafeFree(p_parser->access_chains);

    if (IsNotNull(p_parser->physical_pointer_structs)) {
      SafeFree(p_parser->physical_pointer_structs);
    }
    p_parser->node_count = 0;
  }
}

static SpvReflectResult ParseNodes(SpvReflectPrvParser* p_parser) {
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->spirv_code));

  uint32_t* p_spirv = p_parser->spirv_code;
  uint32_t spirv_word_index = SPIRV_STARTING_WORD_INDEX;

  // Count nodes
  uint32_t node_count = 0;
  while (spirv_word_index < p_parser->spirv_word_count) {
    uint32_t word = p_spirv[spirv_word_index];
    SpvOp op = (SpvOp)(word & 0xFFFF);
    uint32_t node_word_count = (word >> 16) & 0xFFFF;
    if (node_word_count == 0) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_INSTRUCTION;
    }
    if (op == SpvOpAccessChain) {
      ++(p_parser->access_chain_count);
    }
    spirv_word_index += node_word_count;
    ++node_count;
  }

  if (node_count == 0) {
    return SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_EOF;
  }

  // Allocate nodes
  p_parser->node_count = node_count;
  p_parser->nodes = (SpvReflectPrvNode*)calloc(p_parser->node_count, sizeof(*(p_parser->nodes)));
  if (IsNull(p_parser->nodes)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }
  // Mark all nodes with an invalid state
  for (uint32_t i = 0; i < node_count; ++i) {
    p_parser->nodes[i].op = (SpvOp)INVALID_VALUE;
    p_parser->nodes[i].storage_class = (SpvStorageClass)INVALID_VALUE;
    p_parser->nodes[i].decorations.set.value = (uint32_t)INVALID_VALUE;
    p_parser->nodes[i].decorations.binding.value = (uint32_t)INVALID_VALUE;
    p_parser->nodes[i].decorations.location.value = (uint32_t)INVALID_VALUE;
    p_parser->nodes[i].decorations.component.value = (uint32_t)INVALID_VALUE;
    p_parser->nodes[i].decorations.offset.value = (uint32_t)INVALID_VALUE;
    p_parser->nodes[i].decorations.uav_counter_buffer.value = (uint32_t)INVALID_VALUE;
    p_parser->nodes[i].decorations.spec_id = (uint32_t)INVALID_VALUE;
    p_parser->nodes[i].decorations.built_in = (SpvBuiltIn)INVALID_VALUE;
  }
  // Mark source file id node
  p_parser->source_file_id = (uint32_t)INVALID_VALUE;
  p_parser->source_embedded = NULL;

  // Function node
  uint32_t function_node = (uint32_t)INVALID_VALUE;

  // Allocate access chain
  if (p_parser->access_chain_count > 0) {
    p_parser->access_chains = (SpvReflectPrvAccessChain*)calloc(p_parser->access_chain_count, sizeof(*(p_parser->access_chains)));
    if (IsNull(p_parser->access_chains)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }

  // Parse nodes
  uint32_t node_index = 0;
  uint32_t access_chain_index = 0;
  spirv_word_index = SPIRV_STARTING_WORD_INDEX;
  while (spirv_word_index < p_parser->spirv_word_count) {
    uint32_t word = p_spirv[spirv_word_index];
    SpvOp op = (SpvOp)(word & 0xFFFF);
    uint32_t node_word_count = (word >> 16) & 0xFFFF;

    SpvReflectPrvNode* p_node = &(p_parser->nodes[node_index]);
    p_node->op = op;
    p_node->word_offset = spirv_word_index;
    p_node->word_count = node_word_count;

    switch (p_node->op) {
      default:
        break;

      case SpvOpString: {
        ++(p_parser->string_count);
      } break;

      case SpvOpSource: {
        CHECKED_READU32_CAST(p_parser, p_node->word_offset + 1, SpvSourceLanguage, p_parser->source_language);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_parser->source_language_version);
        if (p_node->word_count >= 4) {
          CHECKED_READU32(p_parser, p_node->word_offset + 3, p_parser->source_file_id);
        }
        if (p_node->word_count >= 5) {
          const char* p_source = (const char*)(p_parser->spirv_code + p_node->word_offset + 4);

          const size_t source_len = strlen(p_source);
          char* p_source_temp = (char*)calloc(source_len + 1, sizeof(char));

          if (IsNull(p_source_temp)) {
            return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
          }

#ifdef _WIN32
          strcpy_s(p_source_temp, source_len + 1, p_source);
#else
          strcpy(p_source_temp, p_source);
#endif

          SafeFree(p_parser->source_embedded);
          p_parser->source_embedded = p_source_temp;
        }
      } break;

      case SpvOpSourceContinued: {
        const char* p_source = (const char*)(p_parser->spirv_code + p_node->word_offset + 1);

        const size_t source_len = strlen(p_source);
        const size_t embedded_source_len = strlen(p_parser->source_embedded);
        char* p_continued_source = (char*)calloc(source_len + embedded_source_len + 1, sizeof(char));

        if (IsNull(p_continued_source)) {
          return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
        }

#ifdef _WIN32
        strcpy_s(p_continued_source, embedded_source_len + 1, p_parser->source_embedded);
        strcat_s(p_continued_source, embedded_source_len + source_len + 1, p_source);
#else
        strcpy(p_continued_source, p_parser->source_embedded);
        strcat(p_continued_source, p_source);
#endif

        SafeFree(p_parser->source_embedded);
        p_parser->source_embedded = p_continued_source;
      } break;

      case SpvOpEntryPoint: {
        ++(p_parser->entry_point_count);
      } break;

      case SpvOpCapability: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->capability);
        ++(p_parser->capability_count);
      } break;

      case SpvOpName:
      case SpvOpMemberName: {
        uint32_t member_offset = (p_node->op == SpvOpMemberName) ? 1 : 0;
        uint32_t name_start = p_node->word_offset + member_offset + 2;
        p_node->name = (const char*)(p_parser->spirv_code + name_start);
      } break;

      case SpvOpTypeStruct: {
        p_node->member_count = p_node->word_count - 2;
        FALLTHROUGH;
      }  // Fall through

      // This is all the rest of OpType* that need to be tracked
      // Possible new extensions might expose new type, will need to be added
      // here
      case SpvOpTypeVoid:
      case SpvOpTypeBool:
      case SpvOpTypeInt:
      case SpvOpTypeFloat:
      case SpvOpTypeVector:
      case SpvOpTypeMatrix:
      case SpvOpTypeSampler:
      case SpvOpTypeOpaque:
      case SpvOpTypeFunction:
      case SpvOpTypeEvent:
      case SpvOpTypeDeviceEvent:
      case SpvOpTypeReserveId:
      case SpvOpTypeQueue:
      case SpvOpTypePipe:
      case SpvOpTypeAccelerationStructureKHR:
      case SpvOpTypeRayQueryKHR:
      case SpvOpTypeHitObjectNV:
      case SpvOpTypeCooperativeMatrixNV:
      case SpvOpTypeCooperativeMatrixKHR: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_id);
        p_node->is_type = true;
      } break;

      case SpvOpTypeImage: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->image_traits.sampled_type_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 3, p_node->image_traits.dim);
        CHECKED_READU32(p_parser, p_node->word_offset + 4, p_node->image_traits.depth);
        CHECKED_READU32(p_parser, p_node->word_offset + 5, p_node->image_traits.arrayed);
        CHECKED_READU32(p_parser, p_node->word_offset + 6, p_node->image_traits.ms);
        CHECKED_READU32(p_parser, p_node->word_offset + 7, p_node->image_traits.sampled);
        CHECKED_READU32(p_parser, p_node->word_offset + 8, p_node->image_traits.image_format);
        p_node->is_type = true;
      } break;

      case SpvOpTypeSampledImage: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->image_type_id);
        p_node->is_type = true;
      } break;

      case SpvOpTypeArray: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->array_traits.element_type_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 3, p_node->array_traits.length_id);
        p_node->is_type = true;
      } break;

      case SpvOpTypeRuntimeArray: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->array_traits.element_type_id);
        p_node->is_type = true;
      } break;

      case SpvOpTypePointer: {
        uint32_t result_id;
        CHECKED_READU32(p_parser, p_node->word_offset + 1, result_id);
        // Look for forward pointer. Clear result id if found
        SpvReflectPrvNode* p_fwd_node = FindNode(p_parser, result_id);
        if (p_fwd_node) {
          p_fwd_node->result_id = 0;
        }
        // Register pointer type
        p_node->result_id = result_id;
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->storage_class);
        CHECKED_READU32(p_parser, p_node->word_offset + 3, p_node->type_id);
        p_node->is_type = true;
      } break;

      case SpvOpTypeForwardPointer: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->storage_class);
        p_node->is_type = true;
      } break;

      case SpvOpConstantTrue:
      case SpvOpConstantFalse:
      case SpvOpConstant:
      case SpvOpConstantComposite:
      case SpvOpConstantSampler:
      case SpvOpConstantNull: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_type_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->result_id);
      } break;

      case SpvOpSpecConstantTrue:
      case SpvOpSpecConstantFalse:
      case SpvOpSpecConstant:
      case SpvOpSpecConstantComposite:
      case SpvOpSpecConstantOp: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_type_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->result_id);
      } break;

      case SpvOpVariable: {
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->type_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->result_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 3, p_node->storage_class);
      } break;

      case SpvOpLoad: {
        // Only load enough so OpDecorate can reference the node, skip the remaining operands.
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_type_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->result_id);
      } break;

      case SpvOpAccessChain: {
        SpvReflectPrvAccessChain* p_access_chain = &(p_parser->access_chains[access_chain_index]);
        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_access_chain->result_type_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_access_chain->result_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 3, p_access_chain->base_id);
        //
        // SPIRV_ACCESS_CHAIN_INDEX_OFFSET (4) is the number of words up until the first index:
        //   [Node, Result Type Id, Result Id, Base Id, <Indexes>]
        //
        p_access_chain->index_count = (node_word_count - SPIRV_ACCESS_CHAIN_INDEX_OFFSET);
        if (p_access_chain->index_count > 0) {
          p_access_chain->indexes = (uint32_t*)calloc(p_access_chain->index_count, sizeof(*(p_access_chain->indexes)));
          if (IsNull(p_access_chain->indexes)) {
            return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
          }
          // Parse any index values for access chain
          for (uint32_t index_index = 0; index_index < p_access_chain->index_count; ++index_index) {
            // Read index id
            uint32_t index_id = 0;
            CHECKED_READU32(p_parser, p_node->word_offset + SPIRV_ACCESS_CHAIN_INDEX_OFFSET + index_index, index_id);
            // Find OpConstant node that contains index value
            SpvReflectPrvNode* p_index_value_node = FindNode(p_parser, index_id);
            if ((p_index_value_node != NULL) &&
                (p_index_value_node->op == SpvOpConstant || p_index_value_node->op == SpvOpSpecConstant)) {
              // Read index value
              uint32_t index_value = UINT32_MAX;
              CHECKED_READU32(p_parser, p_index_value_node->word_offset + 3, index_value);
              assert(index_value != UINT32_MAX);
              // Write index value to array
              p_access_chain->indexes[index_index] = index_value;
            }
          }
        }
        ++access_chain_index;
      } break;

      case SpvOpFunction: {
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->result_id);
        // Count function definitions, not function declarations.  To determine
        // the difference, set an in-function variable, and then if an OpLabel
        // is reached before the end of the function increment the function
        // count.
        function_node = node_index;
      } break;

      case SpvOpLabel: {
        if (function_node != (uint32_t)INVALID_VALUE) {
          SpvReflectPrvNode* p_func_node = &(p_parser->nodes[function_node]);
          CHECKED_READU32(p_parser, p_func_node->word_offset + 2, p_func_node->result_id);
          ++(p_parser->function_count);
        }
        FALLTHROUGH;
      }  // Fall through

      case SpvOpFunctionEnd: {
        function_node = (uint32_t)INVALID_VALUE;
      } break;
      case SpvOpFunctionParameter: {
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->result_id);
      } break;
      case SpvOpBitcast:
      case SpvOpShiftRightLogical:
      case SpvOpIAdd:
      case SpvOpISub:
      case SpvOpIMul:
      case SpvOpUDiv:
      case SpvOpSDiv: {
        CHECKED_READU32(p_parser, p_node->word_offset + 2, p_node->result_id);
      } break;
    }

    if (p_node->is_type) {
      ++(p_parser->type_count);
    }

    spirv_word_index += node_word_count;
    ++node_index;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseStrings(SpvReflectPrvParser* p_parser) {
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->spirv_code));
  assert(IsNotNull(p_parser->nodes));

  // Early out
  if (p_parser->string_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  if (IsNotNull(p_parser) && IsNotNull(p_parser->spirv_code) && IsNotNull(p_parser->nodes)) {
    // Allocate string storage
    p_parser->strings = (SpvReflectPrvString*)calloc(p_parser->string_count, sizeof(*(p_parser->strings)));

    uint32_t string_index = 0;
    for (size_t i = 0; i < p_parser->node_count; ++i) {
      SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
      if (p_node->op != SpvOpString) {
        continue;
      }

      // Paranoid check against string count
      assert(string_index < p_parser->string_count);
      if (string_index >= p_parser->string_count) {
        return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
      }

      // Result id
      SpvReflectPrvString* p_string = &(p_parser->strings[string_index]);
      CHECKED_READU32(p_parser, p_node->word_offset + 1, p_string->result_id);

      // String
      uint32_t string_start = p_node->word_offset + 2;
      p_string->string = (const char*)(p_parser->spirv_code + string_start);

      // Increment string index
      ++string_index;
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseSource(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->spirv_code));

  if (IsNotNull(p_parser) && IsNotNull(p_parser->spirv_code)) {
    // Source file
    if (IsNotNull(p_parser->strings)) {
      for (uint32_t i = 0; i < p_parser->string_count; ++i) {
        SpvReflectPrvString* p_string = &(p_parser->strings[i]);
        if (p_string->result_id == p_parser->source_file_id) {
          p_module->source_file = p_string->string;
          break;
        }
      }
    }

    // Source code
    if (IsNotNull(p_parser->source_embedded)) {
      const size_t source_len = strlen(p_parser->source_embedded);
      char* p_source = (char*)calloc(source_len + 1, sizeof(char));

      if (IsNull(p_source)) {
        return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
      }

#ifdef _WIN32
      strcpy_s(p_source, source_len + 1, p_parser->source_embedded);
#else
      strcpy(p_source, p_parser->source_embedded);
#endif

      p_module->source_source = p_source;
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseFunction(SpvReflectPrvParser* p_parser, SpvReflectPrvNode* p_func_node, SpvReflectPrvFunction* p_func,
                                      size_t first_label_index) {
  p_func->id = p_func_node->result_id;

  p_func->callee_count = 0;
  p_func->accessed_variable_count = 0;

  // First get count to know how much to allocate
  for (size_t i = first_label_index; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if (p_node->op == SpvOpFunctionEnd) {
      break;
    }
    switch (p_node->op) {
      case SpvOpFunctionCall: {
        ++(p_func->callee_count);
      } break;
      case SpvOpLoad:
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpPtrAccessChain:
      case SpvOpArrayLength:
      case SpvOpGenericPtrMemSemantics:
      case SpvOpInBoundsPtrAccessChain:
      case SpvOpStore:
      case SpvOpImageTexelPointer: {
        ++(p_func->accessed_variable_count);
      } break;
      case SpvOpCopyMemory:
      case SpvOpCopyMemorySized: {
        p_func->accessed_variable_count += 2;
      } break;
      default:
        break;
    }
  }

  if (p_func->callee_count > 0) {
    p_func->callees = (uint32_t*)calloc(p_func->callee_count, sizeof(*(p_func->callees)));
    if (IsNull(p_func->callees)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }

  if (p_func->accessed_variable_count > 0) {
    p_func->accessed_variables =
        (SpvReflectPrvAccessedVariable*)calloc(p_func->accessed_variable_count, sizeof(*(p_func->accessed_variables)));
    if (IsNull(p_func->accessed_variables)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }

  p_func->callee_count = 0;
  p_func->accessed_variable_count = 0;
  // Now have allocation, fill in values
  for (size_t i = first_label_index; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if (p_node->op == SpvOpFunctionEnd) {
      break;
    }
    switch (p_node->op) {
      case SpvOpFunctionCall: {
        CHECKED_READU32(p_parser, p_node->word_offset + 3, p_func->callees[p_func->callee_count]);
        (++p_func->callee_count);
      } break;
      case SpvOpLoad:
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpPtrAccessChain:
      case SpvOpArrayLength:
      case SpvOpGenericPtrMemSemantics:
      case SpvOpInBoundsPtrAccessChain:
      case SpvOpImageTexelPointer: {
        const uint32_t result_index = p_node->word_offset + 2;
        const uint32_t ptr_index = p_node->word_offset + 3;
        SpvReflectPrvAccessedVariable* access_ptr = &p_func->accessed_variables[p_func->accessed_variable_count];

        access_ptr->p_node = p_node;
        // Need to track Result ID as not sure there has been any memory access through here yet
        CHECKED_READU32(p_parser, result_index, access_ptr->result_id);
        CHECKED_READU32(p_parser, ptr_index, access_ptr->variable_ptr);
        (++p_func->accessed_variable_count);
      } break;
      case SpvOpStore: {
        const uint32_t result_index = p_node->word_offset + 2;
        CHECKED_READU32(p_parser, result_index, p_func->accessed_variables[p_func->accessed_variable_count].variable_ptr);
        p_func->accessed_variables[p_func->accessed_variable_count].p_node = p_node;
        (++p_func->accessed_variable_count);
      } break;
      case SpvOpCopyMemory:
      case SpvOpCopyMemorySized: {
        // There is no result_id or node, being zero is same as being invalid
        CHECKED_READU32(p_parser, p_node->word_offset + 1,
                        p_func->accessed_variables[p_func->accessed_variable_count].variable_ptr);
        (++p_func->accessed_variable_count);
        CHECKED_READU32(p_parser, p_node->word_offset + 2,
                        p_func->accessed_variables[p_func->accessed_variable_count].variable_ptr);
        (++p_func->accessed_variable_count);
      } break;
      default:
        break;
    }
  }

  if (p_func->callee_count > 0) {
    qsort(p_func->callees, p_func->callee_count, sizeof(*(p_func->callees)), SortCompareUint32);
  }
  p_func->callee_count = (uint32_t)DedupSortedUint32(p_func->callees, p_func->callee_count);

  if (p_func->accessed_variable_count > 0) {
    qsort(p_func->accessed_variables, p_func->accessed_variable_count, sizeof(*(p_func->accessed_variables)),
          SortCompareAccessedVariable);
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static int SortCompareFunctions(const void* a, const void* b) {
  const SpvReflectPrvFunction* af = (const SpvReflectPrvFunction*)a;
  const SpvReflectPrvFunction* bf = (const SpvReflectPrvFunction*)b;
  return (int)af->id - (int)bf->id;
}

static SpvReflectResult ParseFunctions(SpvReflectPrvParser* p_parser) {
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->spirv_code));
  assert(IsNotNull(p_parser->nodes));

  if (IsNotNull(p_parser) && IsNotNull(p_parser->spirv_code) && IsNotNull(p_parser->nodes)) {
    if (p_parser->function_count == 0) {
      return SPV_REFLECT_RESULT_SUCCESS;
    }

    p_parser->functions = (SpvReflectPrvFunction*)calloc(p_parser->function_count, sizeof(*(p_parser->functions)));
    if (IsNull(p_parser->functions)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }

    size_t function_index = 0;
    for (size_t i = 0; i < p_parser->node_count; ++i) {
      SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
      if (p_node->op != SpvOpFunction) {
        continue;
      }

      // Skip over function declarations that aren't definitions
      bool func_definition = false;
      // Intentionally reuse i to avoid iterating over these nodes more than
      // once
      for (; i < p_parser->node_count; ++i) {
        if (p_parser->nodes[i].op == SpvOpLabel) {
          func_definition = true;
          break;
        }
        if (p_parser->nodes[i].op == SpvOpFunctionEnd) {
          break;
        }
      }
      if (!func_definition) {
        continue;
      }

      SpvReflectPrvFunction* p_function = &(p_parser->functions[function_index]);

      SpvReflectResult result = ParseFunction(p_parser, p_node, p_function, i);
      if (result != SPV_REFLECT_RESULT_SUCCESS) {
        return result;
      }

      ++function_index;
    }

    qsort(p_parser->functions, p_parser->function_count, sizeof(*(p_parser->functions)), SortCompareFunctions);

    // Once they're sorted, link the functions with pointers to improve graph
    // traversal efficiency
    for (size_t i = 0; i < p_parser->function_count; ++i) {
      SpvReflectPrvFunction* p_func = &(p_parser->functions[i]);
      if (p_func->callee_count == 0) {
        continue;
      }
      p_func->callee_ptrs = (SpvReflectPrvFunction**)calloc(p_func->callee_count, sizeof(*(p_func->callee_ptrs)));
      for (size_t j = 0, k = 0; j < p_func->callee_count; ++j) {
        while (p_parser->functions[k].id != p_func->callees[j]) {
          ++k;
          if (k >= p_parser->function_count) {
            // Invalid called function ID somewhere
            return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
          }
        }
        p_func->callee_ptrs[j] = &(p_parser->functions[k]);
      }
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseMemberCounts(SpvReflectPrvParser* p_parser) {
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->spirv_code));
  assert(IsNotNull(p_parser->nodes));

  if (IsNotNull(p_parser) && IsNotNull(p_parser->spirv_code) && IsNotNull(p_parser->nodes)) {
    for (size_t i = 0; i < p_parser->node_count; ++i) {
      SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
      if ((p_node->op != SpvOpMemberName) && (p_node->op != SpvOpMemberDecorate)) {
        continue;
      }

      uint32_t target_id = 0;
      uint32_t member_index = (uint32_t)INVALID_VALUE;
      CHECKED_READU32(p_parser, p_node->word_offset + 1, target_id);
      CHECKED_READU32(p_parser, p_node->word_offset + 2, member_index);
      SpvReflectPrvNode* p_target_node = FindNode(p_parser, target_id);
      // Not all nodes get parsed, so FindNode returning NULL is expected.
      if (IsNull(p_target_node)) {
        continue;
      }

      if (member_index == INVALID_VALUE) {
        return SPV_REFLECT_RESULT_ERROR_RANGE_EXCEEDED;
      }

      p_target_node->member_count = Max(p_target_node->member_count, member_index + 1);
    }

    for (uint32_t i = 0; i < p_parser->node_count; ++i) {
      SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
      if (p_node->member_count == 0) {
        continue;
      }

      p_node->member_names = (const char**)calloc(p_node->member_count, sizeof(*(p_node->member_names)));
      if (IsNull(p_node->member_names)) {
        return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
      }

      p_node->member_decorations = (SpvReflectPrvDecorations*)calloc(p_node->member_count, sizeof(*(p_node->member_decorations)));
      if (IsNull(p_node->member_decorations)) {
        return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
      }
    }
  }
  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseNames(SpvReflectPrvParser* p_parser) {
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->spirv_code));
  assert(IsNotNull(p_parser->nodes));

  if (IsNotNull(p_parser) && IsNotNull(p_parser->spirv_code) && IsNotNull(p_parser->nodes)) {
    for (size_t i = 0; i < p_parser->node_count; ++i) {
      SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
      if ((p_node->op != SpvOpName) && (p_node->op != SpvOpMemberName)) {
        continue;
      }

      uint32_t target_id = 0;
      CHECKED_READU32(p_parser, p_node->word_offset + 1, target_id);
      SpvReflectPrvNode* p_target_node = FindNode(p_parser, target_id);
      // Not all nodes get parsed, so FindNode returning NULL is expected.
      if (IsNull(p_target_node)) {
        continue;
      }

      const char** pp_target_name = &(p_target_node->name);
      if (p_node->op == SpvOpMemberName) {
        uint32_t member_index = UINT32_MAX;
        CHECKED_READU32(p_parser, p_node->word_offset + 2, member_index);
        pp_target_name = &(p_target_node->member_names[member_index]);
      }

      *pp_target_name = p_node->name;
    }
  }
  return SPV_REFLECT_RESULT_SUCCESS;
}

// Returns true if user_type matches pattern or if user_type begins with pattern and the next character is ':'
// For example, UserTypeMatches("rwbuffer", "rwbuffer") will be true, UserTypeMatches("rwbuffer", "rwbuffer:<S>") will be true, and
// UserTypeMatches("rwbuffer", "rwbufferfoo") will be false.
static bool UserTypeMatches(const char* user_type, const char* pattern) {
  const size_t pattern_length = strlen(pattern);
  if (strncmp(user_type, pattern, pattern_length) == 0) {
    if (user_type[pattern_length] == ':' || user_type[pattern_length] == '\0') {
      return true;
    }
  }
  return false;
}

static SpvReflectResult ParseDecorations(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  uint32_t spec_constant_count = 0;
  for (uint32_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);

    if ((p_node->op != SpvOpDecorate) && (p_node->op != SpvOpMemberDecorate) && (p_node->op != SpvOpDecorateId) &&
        (p_node->op != SpvOpDecorateString) && (p_node->op != SpvOpMemberDecorateString)) {
      continue;
    }

    // Need to adjust the read offset if this is a member decoration
    uint32_t member_offset = 0;
    if (p_node->op == SpvOpMemberDecorate) {
      member_offset = 1;
    }

    // Get decoration
    uint32_t decoration = (uint32_t)INVALID_VALUE;
    CHECKED_READU32(p_parser, p_node->word_offset + member_offset + 2, decoration);

    // Filter out the decoration that do not affect reflection, otherwise
    // there will be random crashes because the nodes aren't found.
    bool skip = false;
    switch (decoration) {
      default: {
        skip = true;
      } break;
      case SpvDecorationRelaxedPrecision:
      case SpvDecorationBlock:
      case SpvDecorationBufferBlock:
      case SpvDecorationColMajor:
      case SpvDecorationRowMajor:
      case SpvDecorationArrayStride:
      case SpvDecorationMatrixStride:
      case SpvDecorationBuiltIn:
      case SpvDecorationNoPerspective:
      case SpvDecorationFlat:
      case SpvDecorationNonWritable:
      case SpvDecorationNonReadable:
      case SpvDecorationPatch:
      case SpvDecorationPerVertexKHR:
      case SpvDecorationPerTaskNV:
      case SpvDecorationLocation:
      case SpvDecorationComponent:
      case SpvDecorationBinding:
      case SpvDecorationDescriptorSet:
      case SpvDecorationOffset:
      case SpvDecorationInputAttachmentIndex:
      case SpvDecorationSpecId:
      case SpvDecorationWeightTextureQCOM:
      case SpvDecorationBlockMatchTextureQCOM:
      case SpvDecorationUserTypeGOOGLE:
      case SpvDecorationHlslCounterBufferGOOGLE:
      case SpvDecorationHlslSemanticGOOGLE: {
        skip = false;
      } break;
    }
    if (skip) {
      continue;
    }

    // Find target node
    uint32_t target_id = 0;
    CHECKED_READU32(p_parser, p_node->word_offset + 1, target_id);
    SpvReflectPrvNode* p_target_node = FindNode(p_parser, target_id);
    if (IsNull(p_target_node)) {
      if ((p_node->op == (uint32_t)SpvOpDecorate) && (decoration == SpvDecorationRelaxedPrecision)) {
        // Many OPs can be decorated that we don't care about. Ignore those.
        // See https://github.com/KhronosGroup/SPIRV-Reflect/issues/134
        continue;
      }
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }
    // Get decorations
    SpvReflectPrvDecorations* p_target_decorations = &(p_target_node->decorations);
    // Update pointer if this is a member decoration
    if (p_node->op == SpvOpMemberDecorate) {
      uint32_t member_index = (uint32_t)INVALID_VALUE;
      CHECKED_READU32(p_parser, p_node->word_offset + 2, member_index);
      p_target_decorations = &(p_target_node->member_decorations[member_index]);
    }

    switch (decoration) {
      default:
        break;

      case SpvDecorationRelaxedPrecision: {
        p_target_decorations->is_relaxed_precision = true;
      } break;

      case SpvDecorationBlock: {
        p_target_decorations->is_block = true;
      } break;

      case SpvDecorationBufferBlock: {
        p_target_decorations->is_buffer_block = true;
      } break;

      case SpvDecorationColMajor: {
        p_target_decorations->is_column_major = true;
      } break;

      case SpvDecorationRowMajor: {
        p_target_decorations->is_row_major = true;
      } break;

      case SpvDecorationArrayStride: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->array_stride);
      } break;

      case SpvDecorationMatrixStride: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->matrix_stride);
      } break;

      case SpvDecorationBuiltIn: {
        p_target_decorations->is_built_in = true;
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32_CAST(p_parser, word_offset, SpvBuiltIn, p_target_decorations->built_in);
      } break;

      case SpvDecorationNoPerspective: {
        p_target_decorations->is_noperspective = true;
      } break;

      case SpvDecorationFlat: {
        p_target_decorations->is_flat = true;
      } break;

      case SpvDecorationNonWritable: {
        p_target_decorations->is_non_writable = true;
      } break;

      case SpvDecorationNonReadable: {
        p_target_decorations->is_non_readable = true;
      } break;

      case SpvDecorationPatch: {
        p_target_decorations->is_patch = true;
      } break;

      case SpvDecorationPerVertexKHR: {
        p_target_decorations->is_per_vertex = true;
      } break;

      case SpvDecorationPerTaskNV: {
        p_target_decorations->is_per_task = true;
      } break;

      case SpvDecorationLocation: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->location.value);
        p_target_decorations->location.word_offset = word_offset;
      } break;

      case SpvDecorationComponent: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->component.value);
        p_target_decorations->component.word_offset = word_offset;
      } break;

      case SpvDecorationBinding: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->binding.value);
        p_target_decorations->binding.word_offset = word_offset;
      } break;

      case SpvDecorationDescriptorSet: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->set.value);
        p_target_decorations->set.word_offset = word_offset;
      } break;

      case SpvDecorationOffset: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->offset.value);
        p_target_decorations->offset.word_offset = word_offset;
      } break;

      case SpvDecorationInputAttachmentIndex: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->input_attachment_index.value);
        p_target_decorations->input_attachment_index.word_offset = word_offset;
      } break;

      case SpvDecorationSpecId: {
        uint32_t word_offset = p_node->word_offset + member_offset+ 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->spec_id);
        spec_constant_count++;
      } break;

      case SpvDecorationHlslCounterBufferGOOGLE: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        CHECKED_READU32(p_parser, word_offset, p_target_decorations->uav_counter_buffer.value);
        p_target_decorations->uav_counter_buffer.word_offset = word_offset;
      } break;

      case SpvDecorationHlslSemanticGOOGLE: {
        uint32_t word_offset = p_node->word_offset + member_offset + 3;
        p_target_decorations->semantic.value = (const char*)(p_parser->spirv_code + word_offset);
        p_target_decorations->semantic.word_offset = word_offset;
      } break;

      case SpvDecorationWeightTextureQCOM: {
        p_target_decorations->is_weight_texture = true;
      } break;

      case SpvDecorationBlockMatchTextureQCOM: {
        p_target_decorations->is_block_match_texture = true;
      } break;
    }

    if (p_node->op == SpvOpDecorateString && decoration == SpvDecorationUserTypeGOOGLE) {
      uint32_t terminator = 0;
      SpvReflectResult result = ReadStr(p_parser, p_node->word_offset + 3, 0, p_node->word_count, &terminator, NULL);
      if (result != SPV_REFLECT_RESULT_SUCCESS) {
        return result;
      }
      const char* name = (const char*)(p_parser->spirv_code + p_node->word_offset + 3);
      if (UserTypeMatches(name, "cbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_CBUFFER;
      } else if (UserTypeMatches(name, "tbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TBUFFER;
      } else if (UserTypeMatches(name, "appendstructuredbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_APPEND_STRUCTURED_BUFFER;
      } else if (UserTypeMatches(name, "buffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_BUFFER;
      } else if (UserTypeMatches(name, "byteaddressbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_BYTE_ADDRESS_BUFFER;
      } else if (UserTypeMatches(name, "constantbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_CONSTANT_BUFFER;
      } else if (UserTypeMatches(name, "consumestructuredbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_CONSUME_STRUCTURED_BUFFER;
      } else if (UserTypeMatches(name, "inputpatch")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_INPUT_PATCH;
      } else if (UserTypeMatches(name, "outputpatch")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_OUTPUT_PATCH;
      } else if (UserTypeMatches(name, "rasterizerorderedbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RASTERIZER_ORDERED_BUFFER;
      } else if (UserTypeMatches(name, "rasterizerorderedbyteaddressbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RASTERIZER_ORDERED_BYTE_ADDRESS_BUFFER;
      } else if (UserTypeMatches(name, "rasterizerorderedstructuredbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RASTERIZER_ORDERED_STRUCTURED_BUFFER;
      } else if (UserTypeMatches(name, "rasterizerorderedtexture1d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RASTERIZER_ORDERED_TEXTURE_1D;
      } else if (UserTypeMatches(name, "rasterizerorderedtexture1darray")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RASTERIZER_ORDERED_TEXTURE_1D_ARRAY;
      } else if (UserTypeMatches(name, "rasterizerorderedtexture2d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RASTERIZER_ORDERED_TEXTURE_2D;
      } else if (UserTypeMatches(name, "rasterizerorderedtexture2darray")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RASTERIZER_ORDERED_TEXTURE_2D_ARRAY;
      } else if (UserTypeMatches(name, "rasterizerorderedtexture3d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RASTERIZER_ORDERED_TEXTURE_3D;
      } else if (UserTypeMatches(name, "raytracingaccelerationstructure")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RAYTRACING_ACCELERATION_STRUCTURE;
      } else if (UserTypeMatches(name, "rwbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RW_BUFFER;
      } else if (UserTypeMatches(name, "rwbyteaddressbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RW_BYTE_ADDRESS_BUFFER;
      } else if (UserTypeMatches(name, "rwstructuredbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RW_STRUCTURED_BUFFER;
      } else if (UserTypeMatches(name, "rwtexture1d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RW_TEXTURE_1D;
      } else if (UserTypeMatches(name, "rwtexture1darray")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RW_TEXTURE_1D_ARRAY;
      } else if (UserTypeMatches(name, "rwtexture2d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RW_TEXTURE_2D;
      } else if (UserTypeMatches(name, "rwtexture2darray")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RW_TEXTURE_2D_ARRAY;
      } else if (UserTypeMatches(name, "rwtexture3d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_RW_TEXTURE_3D;
      } else if (UserTypeMatches(name, "structuredbuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_STRUCTURED_BUFFER;
      } else if (UserTypeMatches(name, "subpassinput")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_SUBPASS_INPUT;
      } else if (UserTypeMatches(name, "subpassinputms")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_SUBPASS_INPUT_MS;
      } else if (UserTypeMatches(name, "texture1d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_1D;
      } else if (UserTypeMatches(name, "texture1darray")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_1D_ARRAY;
      } else if (UserTypeMatches(name, "texture2d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_2D;
      } else if (UserTypeMatches(name, "texture2darray")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_2D_ARRAY;
      } else if (UserTypeMatches(name, "texture2dms")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_2DMS;
      } else if (UserTypeMatches(name, "texture2dmsarray")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_2DMS_ARRAY;
      } else if (UserTypeMatches(name, "texture3d")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_3D;
      } else if (UserTypeMatches(name, "texturebuffer")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_BUFFER;
      } else if (UserTypeMatches(name, "texturecube")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_CUBE;
      } else if (UserTypeMatches(name, "texturecubearray")) {
        p_target_decorations->user_type = SPV_REFLECT_USER_TYPE_TEXTURE_CUBE_ARRAY;
      }
    }
  }

  if (spec_constant_count > 0) {
    p_module->spec_constants = (SpvReflectSpecializationConstant*)calloc(spec_constant_count, sizeof(*p_module->spec_constants));
    if (IsNull(p_module->spec_constants)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }
  for (uint32_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    const uint32_t count = p_module->spec_constant_count;
    switch(p_node->op) {
      default: continue;
      case SpvOpSpecConstantTrue: {
        p_module->spec_constants[count].constant_type = SPV_REFLECT_SPECIALIZATION_CONSTANT_BOOL;
        p_module->spec_constants[count].default_value.int_bool_value = 1;
      } break;
      case SpvOpSpecConstantFalse: {
        p_module->spec_constants[count].constant_type = SPV_REFLECT_SPECIALIZATION_CONSTANT_BOOL;
        p_module->spec_constants[count].default_value.int_bool_value = 0;
      } break;
      case SpvOpSpecConstant: {
        SpvReflectResult result = SPV_REFLECT_RESULT_SUCCESS;
        uint32_t element_type_id = (uint32_t)INVALID_VALUE;
        uint32_t default_value = 0;
        CHECKED_READU32(p_parser, p_node->word_offset + 1, element_type_id);
        CHECKED_READU32(p_parser, p_node->word_offset + 3, default_value);

        SpvReflectPrvNode* p_next_node = FindNode(p_parser, element_type_id);

        if (p_next_node->op == SpvOpTypeInt) {
          p_module->spec_constants[count].constant_type = SPV_REFLECT_SPECIALIZATION_CONSTANT_INT;
        } else if (p_next_node->op == SpvOpTypeFloat) {
          p_module->spec_constants[count].constant_type = SPV_REFLECT_SPECIALIZATION_CONSTANT_FLOAT;
        } else {
          return SPV_REFLECT_RESULT_ERROR_PARSE_FAILED;
        }

        p_module->spec_constants[count].default_value.int_bool_value = default_value; //bits are the same for int and float
      } break;
    }

    p_module->spec_constants[count].name = p_node->name;
    p_module->spec_constants[count].constant_id = p_node->decorations.spec_id;
    p_module->spec_constants[count].spirv_id = p_node->result_id;

    p_module->spec_constant_count++;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult EnumerateAllUniforms(SpvReflectShaderModule* p_module, size_t* p_uniform_count, uint32_t** pp_uniforms) {
  *p_uniform_count = p_module->descriptor_binding_count;
  if (*p_uniform_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }
  *pp_uniforms = (uint32_t*)calloc(*p_uniform_count, sizeof(**pp_uniforms));

  if (IsNull(*pp_uniforms)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }

  for (size_t i = 0; i < *p_uniform_count; ++i) {
    (*pp_uniforms)[i] = p_module->descriptor_bindings[i].spirv_id;
  }
  qsort(*pp_uniforms, *p_uniform_count, sizeof(**pp_uniforms), SortCompareUint32);
  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseType(SpvReflectPrvParser* p_parser, SpvReflectPrvNode* p_node,
                                  SpvReflectPrvDecorations* p_struct_member_decorations, SpvReflectShaderModule* p_module,
                                  SpvReflectTypeDescription* p_type) {
  SpvReflectResult result = SPV_REFLECT_RESULT_SUCCESS;

  if (p_node->member_count > 0) {
    p_type->struct_type_description = FindType(p_module, p_node->result_id);
    p_type->member_count = p_node->member_count;
    p_type->members = (SpvReflectTypeDescription*)calloc(p_type->member_count, sizeof(*(p_type->members)));
    if (IsNotNull(p_type->members)) {
      // Mark all members types with an invalid state
      for (size_t i = 0; i < p_type->members->member_count; ++i) {
        SpvReflectTypeDescription* p_member_type = &(p_type->members[i]);
        p_member_type->id = (uint32_t)INVALID_VALUE;
        p_member_type->op = (SpvOp)INVALID_VALUE;
        p_member_type->storage_class = (SpvStorageClass)INVALID_VALUE;
      }
    } else {
      result = SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }

  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    // Since the parse descends on type information, these will get overwritten
    // if not guarded against assignment. Only assign if the id is invalid.
    if (p_type->id == INVALID_VALUE) {
      p_type->id = p_node->result_id;
      p_type->op = p_node->op;
      p_type->decoration_flags = 0;
    }
    // Top level types need to pick up decorations from all types below it.
    // Issue and fix here: https://github.com/chaoticbob/SPIRV-Reflect/issues/64
    p_type->decoration_flags = ApplyDecorations(&p_node->decorations);

    switch (p_node->op) {
      default:
        break;
      case SpvOpTypeVoid:
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_VOID;
        break;

      case SpvOpTypeBool:
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_BOOL;
        break;

      case SpvOpTypeInt: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_INT;
        IF_READU32(result, p_parser, p_node->word_offset + 2, p_type->traits.numeric.scalar.width);
        IF_READU32(result, p_parser, p_node->word_offset + 3, p_type->traits.numeric.scalar.signedness);
      } break;

      case SpvOpTypeFloat: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_FLOAT;
        IF_READU32(result, p_parser, p_node->word_offset + 2, p_type->traits.numeric.scalar.width);
      } break;

      case SpvOpTypeVector: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_VECTOR;
        uint32_t component_type_id = (uint32_t)INVALID_VALUE;
        IF_READU32(result, p_parser, p_node->word_offset + 2, component_type_id);
        IF_READU32(result, p_parser, p_node->word_offset + 3, p_type->traits.numeric.vector.component_count);
        // Parse component type
        SpvReflectPrvNode* p_next_node = FindNode(p_parser, component_type_id);
        if (IsNotNull(p_next_node)) {
          result = ParseType(p_parser, p_next_node, NULL, p_module, p_type);
        } else {
          result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
          SPV_REFLECT_ASSERT(false);
        }
      } break;

      case SpvOpTypeMatrix: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_MATRIX;
        uint32_t column_type_id = (uint32_t)INVALID_VALUE;
        IF_READU32(result, p_parser, p_node->word_offset + 2, column_type_id);
        IF_READU32(result, p_parser, p_node->word_offset + 3, p_type->traits.numeric.matrix.column_count);
        SpvReflectPrvNode* p_next_node = FindNode(p_parser, column_type_id);
        if (IsNotNull(p_next_node)) {
          result = ParseType(p_parser, p_next_node, NULL, p_module, p_type);
        } else {
          result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
          SPV_REFLECT_ASSERT(false);
        }
        p_type->traits.numeric.matrix.row_count = p_type->traits.numeric.vector.component_count;
        p_type->traits.numeric.matrix.stride = p_node->decorations.matrix_stride;
        // NOTE: Matrix stride is decorated using OpMemberDecoreate - not OpDecoreate.
        if (IsNotNull(p_struct_member_decorations)) {
          p_type->traits.numeric.matrix.stride = p_struct_member_decorations->matrix_stride;
        }
      } break;

      case SpvOpTypeImage: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_EXTERNAL_IMAGE;
        uint32_t sampled_type_id = (uint32_t)INVALID_VALUE;
        IF_READU32(result, p_parser, p_node->word_offset + 2, sampled_type_id);
        SpvReflectPrvNode* p_next_node = FindNode(p_parser, sampled_type_id);
        if (IsNotNull(p_next_node)) {
          result = ParseType(p_parser, p_next_node, NULL, p_module, p_type);
        } else {
          result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
        }
        IF_READU32_CAST(result, p_parser, p_node->word_offset + 3, SpvDim, p_type->traits.image.dim);
        IF_READU32(result, p_parser, p_node->word_offset + 4, p_type->traits.image.depth);
        IF_READU32(result, p_parser, p_node->word_offset + 5, p_type->traits.image.arrayed);
        IF_READU32(result, p_parser, p_node->word_offset + 6, p_type->traits.image.ms);
        IF_READU32(result, p_parser, p_node->word_offset + 7, p_type->traits.image.sampled);
        IF_READU32_CAST(result, p_parser, p_node->word_offset + 8, SpvImageFormat, p_type->traits.image.image_format);
      } break;

      case SpvOpTypeSampler: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLER;
      } break;

      case SpvOpTypeSampledImage: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLED_IMAGE;
        uint32_t image_type_id = (uint32_t)INVALID_VALUE;
        IF_READU32(result, p_parser, p_node->word_offset + 2, image_type_id);
        SpvReflectPrvNode* p_next_node = FindNode(p_parser, image_type_id);
        if (IsNotNull(p_next_node)) {
          result = ParseType(p_parser, p_next_node, NULL, p_module, p_type);
        } else {
          result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
          SPV_REFLECT_ASSERT(false);
        }
      } break;

      case SpvOpTypeArray: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_ARRAY;
        if (result == SPV_REFLECT_RESULT_SUCCESS) {
          uint32_t element_type_id = (uint32_t)INVALID_VALUE;
          uint32_t length_id = (uint32_t)INVALID_VALUE;
          IF_READU32(result, p_parser, p_node->word_offset + 2, element_type_id);
          IF_READU32(result, p_parser, p_node->word_offset + 3, length_id);
          // NOTE: Array stride is decorated using OpDecorate instead of
          //       OpMemberDecorate, even if the array is apart of a struct.
          p_type->traits.array.stride = p_node->decorations.array_stride;
          // Get length for current dimension
          SpvReflectPrvNode* p_length_node = FindNode(p_parser, length_id);
          if (IsNotNull(p_length_node)) {
            uint32_t dim_index = p_type->traits.array.dims_count;
            uint32_t length = 0;
            IF_READU32(result, p_parser, p_length_node->word_offset + 3, length);
            if (result == SPV_REFLECT_RESULT_SUCCESS) {
              p_type->traits.array.dims[dim_index] = length;
              p_type->traits.array.dims_count += 1;
              p_type->traits.array.spec_constant_op_ids[dim_index] =
                  IsSpecConstant(p_length_node) ? p_length_node->decorations.spec_id : (uint32_t)INVALID_VALUE;
            } else {
              result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
              SPV_REFLECT_ASSERT(false);
            }
            // Parse next dimension or element type
            SpvReflectPrvNode* p_next_node = FindNode(p_parser, element_type_id);
            if (IsNotNull(p_next_node)) {
              result = ParseType(p_parser, p_next_node, NULL, p_module, p_type);
            }
          } else {
            result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
            SPV_REFLECT_ASSERT(false);
          }
        }
      } break;

      case SpvOpTypeRuntimeArray: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_ARRAY;
        uint32_t element_type_id = (uint32_t)INVALID_VALUE;
        IF_READU32(result, p_parser, p_node->word_offset + 2, element_type_id);
        p_type->traits.array.stride = p_node->decorations.array_stride;
        uint32_t dim_index = p_type->traits.array.dims_count;
        p_type->traits.array.dims[dim_index] = (uint32_t)SPV_REFLECT_ARRAY_DIM_RUNTIME;
        p_type->traits.array.spec_constant_op_ids[dim_index] = (uint32_t)INVALID_VALUE;
        p_type->traits.array.dims_count += 1;
        // Parse next dimension or element type
        SpvReflectPrvNode* p_next_node = FindNode(p_parser, element_type_id);
        if (IsNotNull(p_next_node)) {
          result = ParseType(p_parser, p_next_node, NULL, p_module, p_type);
        } else {
          result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
          SPV_REFLECT_ASSERT(false);
        }
      } break;

      case SpvOpTypeStruct: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_STRUCT;
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_EXTERNAL_BLOCK;
        uint32_t word_index = 2;
        uint32_t member_index = 0;
        for (; word_index < p_node->word_count; ++word_index, ++member_index) {
          uint32_t member_id = (uint32_t)INVALID_VALUE;
          IF_READU32(result, p_parser, p_node->word_offset + word_index, member_id);
          // Find member node
          SpvReflectPrvNode* p_member_node = FindNode(p_parser, member_id);
          if (IsNull(p_member_node)) {
            result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
            SPV_REFLECT_ASSERT(false);
            break;
          }

          // Member decorations
          SpvReflectPrvDecorations* p_member_decorations = &p_node->member_decorations[member_index];

          assert(member_index < p_type->member_count);
          // Parse member type
          SpvReflectTypeDescription* p_member_type = &(p_type->members[member_index]);
          p_member_type->id = member_id;
          p_member_type->op = p_member_node->op;
          result = ParseType(p_parser, p_member_node, p_member_decorations, p_module, p_member_type);
          if (result != SPV_REFLECT_RESULT_SUCCESS) {
            break;
          }
          // This looks wrong
          // p_member_type->type_name = p_member_node->name;
          p_member_type->struct_member_name = p_node->member_names[member_index];
        }
      } break;

      case SpvOpTypeOpaque:
        break;

      case SpvOpTypePointer: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_REF;
        IF_READU32_CAST(result, p_parser, p_node->word_offset + 2, SpvStorageClass, p_type->storage_class);

        bool found_recursion = false;
        if (p_type->storage_class == SpvStorageClassPhysicalStorageBuffer) {
          // Need to make sure we haven't started an infinite recursive loop
          for (uint32_t i = 0; i < p_parser->physical_pointer_count; i++) {
            if (p_type->id == p_parser->physical_pointer_check[i]->id) {
              found_recursion = true;
              memcpy(p_type, p_parser->physical_pointer_check[i], sizeof(SpvReflectTypeDescription));
              p_type->copied = 1;
              return SPV_REFLECT_RESULT_SUCCESS;
            }
          }
          if (!found_recursion) {
            p_parser->physical_pointer_struct_count++;
            p_parser->physical_pointer_check[p_parser->physical_pointer_count] = p_type;
            p_parser->physical_pointer_count++;
            if (p_parser->physical_pointer_count >= MAX_RECURSIVE_PHYSICAL_POINTER_CHECK) {
              return SPV_REFLECT_RESULT_ERROR_SPIRV_MAX_RECURSIVE_EXCEEDED;
            }
          }
        }

        // Parse type
        SpvReflectPrvNode* p_next_node = FindNode(p_parser, p_node->type_id);
        if (IsNull(p_next_node)) {
          result = SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
          SPV_REFLECT_ASSERT(false);
        } else if (!found_recursion) {
          if (p_next_node->op == SpvOpTypeStruct) {
            p_type->struct_type_description = FindType(p_module, p_next_node->result_id);
          }

          result = ParseType(p_parser, p_next_node, NULL, p_module, p_type);
        }
      } break;

      case SpvOpTypeAccelerationStructureKHR: {
        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE;
      } break;
    }

    if (result == SPV_REFLECT_RESULT_SUCCESS) {
      // Names get assigned on the way down. Guard against names
      // get overwritten on the way up.
      if (IsNull(p_type->type_name)) {
        p_type->type_name = p_node->name;
      }
    }
  }

  return result;
}

static SpvReflectResult ParseTypes(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  if (p_parser->type_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  p_module->_internal->type_description_count = p_parser->type_count;
  p_module->_internal->type_descriptions = (SpvReflectTypeDescription*)calloc(p_module->_internal->type_description_count,
                                                                              sizeof(*(p_module->_internal->type_descriptions)));
  if (IsNull(p_module->_internal->type_descriptions)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }

  // Mark all types with an invalid state
  for (size_t i = 0; i < p_module->_internal->type_description_count; ++i) {
    SpvReflectTypeDescription* p_type = &(p_module->_internal->type_descriptions[i]);
    p_type->id = (uint32_t)INVALID_VALUE;
    p_type->op = (SpvOp)INVALID_VALUE;
    p_type->storage_class = (SpvStorageClass)INVALID_VALUE;
  }

  size_t type_index = 0;
  for (size_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if (!p_node->is_type) {
      continue;
    }

    SpvReflectTypeDescription* p_type = &(p_module->_internal->type_descriptions[type_index]);
    p_parser->physical_pointer_count = 0;
    SpvReflectResult result = ParseType(p_parser, p_node, NULL, p_module, p_type);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }
    ++type_index;
  }

  // allocate now and fill in when parsing struct variable later
  if (p_parser->physical_pointer_struct_count > 0) {
    p_parser->physical_pointer_structs = (SpvReflectPrvPhysicalPointerStruct*)calloc(p_parser->physical_pointer_struct_count,
                                                                                     sizeof(*(p_parser->physical_pointer_structs)));
    if (IsNull(p_parser->physical_pointer_structs)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }
  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseCapabilities(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  if (p_parser->capability_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  p_module->capability_count = p_parser->capability_count;
  p_module->capabilities = (SpvReflectCapability*)calloc(p_module->capability_count, sizeof(*(p_module->capabilities)));
  if (IsNull(p_module->capabilities)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }

  // Mark all types with an invalid state
  for (size_t i = 0; i < p_module->capability_count; ++i) {
    SpvReflectCapability* p_cap = &(p_module->capabilities[i]);
    p_cap->value = SpvCapabilityMax;
    p_cap->word_offset = (uint32_t)INVALID_VALUE;
  }

  size_t capability_index = 0;
  for (size_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if (SpvOpCapability != p_node->op) {
      continue;
    }

    SpvReflectCapability* p_cap = &(p_module->capabilities[capability_index]);
    p_cap->value = p_node->capability;
    p_cap->word_offset = p_node->word_offset + 1;
    ++capability_index;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static int SortCompareDescriptorBinding(const void* a, const void* b) {
  const SpvReflectDescriptorBinding* p_elem_a = (const SpvReflectDescriptorBinding*)a;
  const SpvReflectDescriptorBinding* p_elem_b = (const SpvReflectDescriptorBinding*)b;
  int value = (int)(p_elem_a->binding) - (int)(p_elem_b->binding);
  if (value == 0) {
    // use spirv-id as a tiebreaker to ensure a stable ordering, as they're guaranteed
    // unique.
    assert(p_elem_a->spirv_id != p_elem_b->spirv_id);
    value = (int)(p_elem_a->spirv_id) - (int)(p_elem_b->spirv_id);
  }
  return value;
}

static SpvReflectResult ParseDescriptorBindings(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  p_module->descriptor_binding_count = 0;
  for (size_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if ((p_node->op != SpvOpVariable) ||
        ((p_node->storage_class != SpvStorageClassUniform) && (p_node->storage_class != SpvStorageClassStorageBuffer) &&
         (p_node->storage_class != SpvStorageClassUniformConstant))) {
      continue;
    }
    if ((p_node->decorations.set.value == INVALID_VALUE) || (p_node->decorations.binding.value == INVALID_VALUE)) {
      continue;
    }

    p_module->descriptor_binding_count += 1;
  }

  if (p_module->descriptor_binding_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  p_module->descriptor_bindings =
      (SpvReflectDescriptorBinding*)calloc(p_module->descriptor_binding_count, sizeof(*(p_module->descriptor_bindings)));
  if (IsNull(p_module->descriptor_bindings)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }

  // Mark all types with an invalid state
  for (uint32_t descriptor_index = 0; descriptor_index < p_module->descriptor_binding_count; ++descriptor_index) {
    SpvReflectDescriptorBinding* p_descriptor = &(p_module->descriptor_bindings[descriptor_index]);
    p_descriptor->binding = (uint32_t)INVALID_VALUE;
    p_descriptor->input_attachment_index = (uint32_t)INVALID_VALUE;
    p_descriptor->set = (uint32_t)INVALID_VALUE;
    p_descriptor->descriptor_type = (SpvReflectDescriptorType)INVALID_VALUE;
    p_descriptor->uav_counter_id = (uint32_t)INVALID_VALUE;
  }

  size_t descriptor_index = 0;
  for (size_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if ((p_node->op != SpvOpVariable) ||
        ((p_node->storage_class != SpvStorageClassUniform) && (p_node->storage_class != SpvStorageClassStorageBuffer) &&
         (p_node->storage_class != SpvStorageClassUniformConstant))) {
      continue;
    }
    if ((p_node->decorations.set.value == INVALID_VALUE) || (p_node->decorations.binding.value == INVALID_VALUE)) {
      continue;
    }

    SpvReflectTypeDescription* p_type = FindType(p_module, p_node->type_id);
    if (IsNull(p_type)) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }
    // If the type is a pointer, resolve it. We need to retain the storage class
    // from the pointer so that we can use it to deduce deescriptor types.
    SpvStorageClass pointer_storage_class = SpvStorageClassMax;
    if (p_type->op == SpvOpTypePointer) {
      pointer_storage_class = p_type->storage_class;
      // Find the type's node
      SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
      if (IsNull(p_type_node)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
      // Should be the resolved type
      p_type = FindType(p_module, p_type_node->type_id);
      if (IsNull(p_type)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
    }

    SpvReflectDescriptorBinding* p_descriptor = &p_module->descriptor_bindings[descriptor_index];
    p_descriptor->spirv_id = p_node->result_id;
    p_descriptor->name = p_node->name;
    p_descriptor->binding = p_node->decorations.binding.value;
    p_descriptor->input_attachment_index = p_node->decorations.input_attachment_index.value;
    p_descriptor->set = p_node->decorations.set.value;
    p_descriptor->count = 1;
    p_descriptor->uav_counter_id = p_node->decorations.uav_counter_buffer.value;
    p_descriptor->type_description = p_type;
    p_descriptor->decoration_flags = ApplyDecorations(&p_node->decorations);
    p_descriptor->user_type = p_node->decorations.user_type;

    // Flags like non-writable and non-readable are found as member decorations only.
    // If all members have one of those decorations set, promote the decoration up
    // to the whole descriptor.
    const SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
    if (IsNotNull(p_type_node) && p_type_node->member_count) {
      SpvReflectPrvDecorations common_flags = p_type_node->member_decorations[0];

      for (uint32_t m = 1; m < p_type_node->member_count; ++m) {
        common_flags.is_relaxed_precision &= p_type_node->member_decorations[m].is_relaxed_precision;
        common_flags.is_block &= p_type_node->member_decorations[m].is_block;
        common_flags.is_buffer_block &= p_type_node->member_decorations[m].is_buffer_block;
        common_flags.is_row_major &= p_type_node->member_decorations[m].is_row_major;
        common_flags.is_column_major &= p_type_node->member_decorations[m].is_column_major;
        common_flags.is_built_in &= p_type_node->member_decorations[m].is_built_in;
        common_flags.is_noperspective &= p_type_node->member_decorations[m].is_noperspective;
        common_flags.is_flat &= p_type_node->member_decorations[m].is_flat;
        common_flags.is_non_writable &= p_type_node->member_decorations[m].is_non_writable;
        common_flags.is_non_readable &= p_type_node->member_decorations[m].is_non_readable;
        common_flags.is_patch &= p_type_node->member_decorations[m].is_patch;
        common_flags.is_per_vertex &= p_type_node->member_decorations[m].is_per_vertex;
        common_flags.is_per_task &= p_type_node->member_decorations[m].is_per_task;
        common_flags.is_weight_texture &= p_type_node->member_decorations[m].is_weight_texture;
        common_flags.is_block_match_texture &= p_type_node->member_decorations[m].is_block_match_texture;
      }

      p_descriptor->decoration_flags |= ApplyDecorations(&common_flags);
    }

    // If this is in the StorageBuffer storage class, it's for sure a storage
    // buffer descriptor. We need to handle this case earlier because in SPIR-V
    // there are two ways to indicate a storage buffer:
    // 1) Uniform storage class + BufferBlock decoration, or
    // 2) StorageBuffer storage class + Buffer decoration.
    // The 1) way is deprecated since SPIR-V v1.3. But the Buffer decoration is
    // also used together with Uniform storage class to mean uniform buffer..
    // We'll handle the pre-v1.3 cases in ParseDescriptorType().
    if (pointer_storage_class == SpvStorageClassStorageBuffer) {
      p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }

    // Copy image traits
    if ((p_type->type_flags & SPV_REFLECT_TYPE_FLAG_EXTERNAL_MASK) == SPV_REFLECT_TYPE_FLAG_EXTERNAL_IMAGE) {
      memcpy(&p_descriptor->image, &p_type->traits.image, sizeof(p_descriptor->image));
    }

    // This is a workaround for: https://github.com/KhronosGroup/glslang/issues/1096
    {
      const uint32_t resource_mask = SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLED_IMAGE | SPV_REFLECT_TYPE_FLAG_EXTERNAL_IMAGE;
      if ((p_type->type_flags & resource_mask) == resource_mask) {
        memcpy(&p_descriptor->image, &p_type->traits.image, sizeof(p_descriptor->image));
      }
    }

    // Copy array traits
    if (p_type->traits.array.dims_count > 0) {
      p_descriptor->array.dims_count = p_type->traits.array.dims_count;
      for (uint32_t dim_index = 0; dim_index < p_type->traits.array.dims_count; ++dim_index) {
        uint32_t dim_value = p_type->traits.array.dims[dim_index];
        p_descriptor->array.dims[dim_index] = dim_value;
        p_descriptor->count *= dim_value;
      }
    }

    // Count

    p_descriptor->word_offset.binding = p_node->decorations.binding.word_offset;
    p_descriptor->word_offset.set = p_node->decorations.set.word_offset;

    ++descriptor_index;
  }

  if (p_module->descriptor_binding_count > 0) {
    qsort(p_module->descriptor_bindings, p_module->descriptor_binding_count, sizeof(*(p_module->descriptor_bindings)),
          SortCompareDescriptorBinding);
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseDescriptorType(SpvReflectShaderModule* p_module) {
  if (p_module->descriptor_binding_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  for (uint32_t descriptor_index = 0; descriptor_index < p_module->descriptor_binding_count; ++descriptor_index) {
    SpvReflectDescriptorBinding* p_descriptor = &(p_module->descriptor_bindings[descriptor_index]);
    SpvReflectTypeDescription* p_type = p_descriptor->type_description;

    if ((int)p_descriptor->descriptor_type == (int)INVALID_VALUE) {
      switch (p_type->type_flags & SPV_REFLECT_TYPE_FLAG_EXTERNAL_MASK) {
        default:
          assert(false && "unknown type flag");
          break;

        case SPV_REFLECT_TYPE_FLAG_EXTERNAL_IMAGE: {
          if (p_descriptor->image.dim == SpvDimBuffer) {
            switch (p_descriptor->image.sampled) {
              default:
                assert(false && "unknown texel buffer sampled value");
                break;
              case IMAGE_SAMPLED:
                p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                break;
              case IMAGE_STORAGE:
                p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                break;
            }
          } else if (p_descriptor->image.dim == SpvDimSubpassData) {
            p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
          } else {
            switch (p_descriptor->image.sampled) {
              default:
                assert(false && "unknown image sampled value");
                break;
              case IMAGE_SAMPLED:
                p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
              case IMAGE_STORAGE:
                p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            }
          }
        } break;

        case SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLER: {
          p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER;
        } break;

        case (SPV_REFLECT_TYPE_FLAG_EXTERNAL_SAMPLED_IMAGE | SPV_REFLECT_TYPE_FLAG_EXTERNAL_IMAGE): {
          // This is a workaround for: https://github.com/KhronosGroup/glslang/issues/1096
          if (p_descriptor->image.dim == SpvDimBuffer) {
            switch (p_descriptor->image.sampled) {
              default:
                assert(false && "unknown texel buffer sampled value");
                break;
              case IMAGE_SAMPLED:
                p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                break;
              case IMAGE_STORAGE:
                p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                break;
            }
          } else {
            p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
          }
        } break;

        case SPV_REFLECT_TYPE_FLAG_EXTERNAL_BLOCK: {
          if (p_type->decoration_flags & SPV_REFLECT_DECORATION_BLOCK) {
            p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
          } else if (p_type->decoration_flags & SPV_REFLECT_DECORATION_BUFFER_BLOCK) {
            p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          } else {
            assert(false && "unknown struct");
          }
        } break;

        case SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE: {
          p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        } break;
      }
    }

    switch (p_descriptor->descriptor_type) {
      case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_SAMPLER;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        p_descriptor->resource_type = (SpvReflectResourceType)(SPV_REFLECT_RESOURCE_FLAG_SAMPLER | SPV_REFLECT_RESOURCE_FLAG_SRV);
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_SRV;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_UAV;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_SRV;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_UAV;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_CBV;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_CBV;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_UAV;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_UAV;
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        break;
      case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_SRV;
        break;
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseUAVCounterBindings(SpvReflectShaderModule* p_module) {
  char name[MAX_NODE_NAME_LENGTH];
  const char* k_count_tag = "@count";

  for (uint32_t descriptor_index = 0; descriptor_index < p_module->descriptor_binding_count; ++descriptor_index) {
    SpvReflectDescriptorBinding* p_descriptor = &(p_module->descriptor_bindings[descriptor_index]);

    if (p_descriptor->descriptor_type != SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      continue;
    }

    SpvReflectDescriptorBinding* p_counter_descriptor = NULL;
    // Use UAV counter buffer id if present...
    if (p_descriptor->uav_counter_id != UINT32_MAX) {
      for (uint32_t counter_descriptor_index = 0; counter_descriptor_index < p_module->descriptor_binding_count;
           ++counter_descriptor_index) {
        SpvReflectDescriptorBinding* p_test_counter_descriptor = &(p_module->descriptor_bindings[counter_descriptor_index]);
        if (p_test_counter_descriptor->descriptor_type != SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          continue;
        }
        if (p_descriptor->uav_counter_id == p_test_counter_descriptor->spirv_id) {
          p_counter_descriptor = p_test_counter_descriptor;
          break;
        }
      }
    }
    // ...otherwise use old @count convention.
    else {
      const size_t descriptor_name_length = p_descriptor->name ? strlen(p_descriptor->name) : 0;

      memset(name, 0, MAX_NODE_NAME_LENGTH);
      memcpy(name, p_descriptor->name, descriptor_name_length);
#if defined(_WIN32)
      strcat_s(name, MAX_NODE_NAME_LENGTH, k_count_tag);
#else
      strcat(name, k_count_tag);
#endif

      for (uint32_t counter_descriptor_index = 0; counter_descriptor_index < p_module->descriptor_binding_count;
           ++counter_descriptor_index) {
        SpvReflectDescriptorBinding* p_test_counter_descriptor = &(p_module->descriptor_bindings[counter_descriptor_index]);
        if (p_test_counter_descriptor->descriptor_type != SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          continue;
        }
        if (p_test_counter_descriptor->name && strcmp(name, p_test_counter_descriptor->name) == 0) {
          p_counter_descriptor = p_test_counter_descriptor;
          break;
        }
      }
    }

    if (p_counter_descriptor != NULL) {
      p_descriptor->uav_counter_binding = p_counter_descriptor;
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseDescriptorBlockVariable(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module,
                                                     SpvReflectTypeDescription* p_type, SpvReflectBlockVariable* p_var) {
  bool has_non_writable = false;

  if (IsNotNull(p_type->members) && (p_type->member_count > 0)) {
    p_var->member_count = p_type->member_count;
    p_var->members = (SpvReflectBlockVariable*)calloc(p_var->member_count, sizeof(*p_var->members));
    if (IsNull(p_var->members)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }

    SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
    if (IsNull(p_type_node)) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }
    // Resolve to element type if current type is array or run time array
    while (p_type_node->op == SpvOpTypeArray || p_type_node->op == SpvOpTypeRuntimeArray) {
      if (p_type_node->op == SpvOpTypeArray) {
        p_type_node = FindNode(p_parser, p_type_node->array_traits.element_type_id);
      } else {
        // Element type description
        SpvReflectTypeDescription* p_type_temp = FindType(p_module, p_type_node->array_traits.element_type_id);
        if (IsNull(p_type_temp)) {
          return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
        }
        // Element type node
        p_type_node = FindNode(p_parser, p_type_temp->id);
      }
      if (IsNull(p_type_node)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
    }

    // Parse members
    for (uint32_t member_index = 0; member_index < p_type->member_count; ++member_index) {
      SpvReflectTypeDescription* p_member_type = &p_type->members[member_index];
      SpvReflectBlockVariable* p_member_var = &p_var->members[member_index];
      // If pointer type, treat like reference and resolve to pointee type
      SpvReflectTypeDescription* p_member_ptr_type = 0;
      bool found_recursion = false;

      if ((p_member_type->storage_class == SpvStorageClassPhysicalStorageBuffer) &&
          (p_member_type->type_flags & SPV_REFLECT_TYPE_FLAG_REF)) {
        // Remember the original type
        p_member_ptr_type = p_member_type;

        // strip array
        if (p_member_type->op == SpvOpTypeArray) {
          SpvReflectPrvNode* p_node = FindNode(p_parser, p_member_type->id);
          if (p_node == NULL) {
            return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
          }
          uint32_t element_type_id = p_node->array_traits.element_type_id;
          p_member_type = FindType(p_module, element_type_id);
          if (p_member_type == NULL) {
            return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
          }
        }

        // Need to make sure we haven't started an infinite recursive loop
        for (uint32_t i = 0; i < p_parser->physical_pointer_count; i++) {
          if (p_member_type->id == p_parser->physical_pointer_check[i]->id) {
            found_recursion = true;
            break;  // still need to fill in p_member_type values
          }
        }
        if (!found_recursion) {
          uint32_t struct_id = FindType(p_module, p_member_type->id)->struct_type_description->id;
          p_parser->physical_pointer_structs[p_parser->physical_pointer_struct_count].struct_id = struct_id;
          p_parser->physical_pointer_structs[p_parser->physical_pointer_struct_count].p_var = p_member_var;
          p_parser->physical_pointer_struct_count++;

          p_parser->physical_pointer_check[p_parser->physical_pointer_count] = p_member_type;
          p_parser->physical_pointer_count++;
          if (p_parser->physical_pointer_count >= MAX_RECURSIVE_PHYSICAL_POINTER_CHECK) {
            return SPV_REFLECT_RESULT_ERROR_SPIRV_MAX_RECURSIVE_EXCEEDED;
          }
        }

        SpvReflectPrvNode* p_member_type_node = FindNode(p_parser, p_member_type->id);
        if (IsNull(p_member_type_node)) {
          return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
        }
        // Should be the pointee type
        p_member_type = FindType(p_module, p_member_type_node->type_id);
        if (IsNull(p_member_type)) {
          return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
        }
      }
      bool is_struct = (p_member_type->type_flags & SPV_REFLECT_TYPE_FLAG_STRUCT) == SPV_REFLECT_TYPE_FLAG_STRUCT;
      if (is_struct) {
        if (!found_recursion) {
          SpvReflectResult result = ParseDescriptorBlockVariable(p_parser, p_module, p_member_type, p_member_var);
          if (result != SPV_REFLECT_RESULT_SUCCESS) {
            return result;
          }
        } else {
          // if 2 member of structs are same PhysicalPointer type, copy the
          // members values that aren't found skipping the recursion call
          for (uint32_t i = 0; i < p_parser->physical_pointer_struct_count; i++) {
            if (p_parser->physical_pointer_structs[i].struct_id == p_member_type->id) {
              p_member_var->members = p_parser->physical_pointer_structs[i].p_var->members;
              p_member_var->member_count = p_parser->physical_pointer_structs[i].p_var->member_count;
              // Set here as it is the first time we need to walk down structs
              p_member_var->flags |= SPV_REFLECT_VARIABLE_FLAGS_PHYSICAL_POINTER_COPY;
            }
          }
        }
      }

      if (p_type_node->storage_class == SpvStorageClassPhysicalStorageBuffer && !p_type_node->member_names) {
        // TODO 212 - If a buffer ref has an array of itself, all members are null
        continue;
      }

      p_member_var->name = p_type_node->member_names[member_index];
      p_member_var->offset = p_type_node->member_decorations[member_index].offset.value;
      p_member_var->decoration_flags = ApplyDecorations(&p_type_node->member_decorations[member_index]);
      p_member_var->flags |= SPV_REFLECT_VARIABLE_FLAGS_UNUSED;
      if (!has_non_writable && (p_member_var->decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE)) {
        has_non_writable = true;
      }
      ApplyNumericTraits(p_member_type, &p_member_var->numeric);
      if (p_member_type->op == SpvOpTypeArray) {
        ApplyArrayTraits(p_member_type, &p_member_var->array);
      }

      p_member_var->word_offset.offset = p_type_node->member_decorations[member_index].offset.word_offset;
      p_member_var->type_description = p_member_ptr_type ? p_member_ptr_type : p_member_type;
    }
  }

  p_var->name = p_type->type_name;
  p_var->type_description = p_type;
  if (has_non_writable) {
    p_var->decoration_flags |= SPV_REFLECT_DECORATION_NON_WRITABLE;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static uint32_t GetPhysicalPointerStructSize(SpvReflectPrvParser* p_parser, uint32_t id) {
  for (uint32_t i = 0; i < p_parser->physical_pointer_struct_count; i++) {
    if (p_parser->physical_pointer_structs[i].struct_id == id) {
      return p_parser->physical_pointer_structs[i].p_var->size;
    }
  }
  return 0;
}

static SpvReflectResult ParseDescriptorBlockVariableSizes(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module,
                                                          bool is_parent_root, bool is_parent_aos, bool is_parent_rta,
                                                          SpvReflectBlockVariable* p_var) {
  if (p_var->member_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  bool is_parent_ref = p_var->type_description->op == SpvOpTypePointer;

  // Absolute offsets
  for (uint32_t member_index = 0; member_index < p_var->member_count; ++member_index) {
    SpvReflectBlockVariable* p_member_var = &p_var->members[member_index];
    if (is_parent_root) {
      p_member_var->absolute_offset = p_member_var->offset;
    } else {
      p_member_var->absolute_offset =
          is_parent_aos ? 0 : (is_parent_ref ? p_member_var->offset : p_member_var->offset + p_var->absolute_offset);
    }
  }

  // Size
  for (uint32_t member_index = 0; member_index < p_var->member_count; ++member_index) {
    SpvReflectBlockVariable* p_member_var = &p_var->members[member_index];
    SpvReflectTypeDescription* p_member_type = p_member_var->type_description;

    if (!p_member_type) {
      // TODO 212 - If a buffer ref has an array of itself, all members are null
      continue;
    }
    switch (p_member_type->op) {
      case SpvOpTypeBool: {
        p_member_var->size = SPIRV_WORD_SIZE;
      } break;

      case SpvOpTypeInt:
      case SpvOpTypeFloat: {
        p_member_var->size = p_member_type->traits.numeric.scalar.width / SPIRV_BYTE_WIDTH;
      } break;

      case SpvOpTypeVector: {
        uint32_t size =
            p_member_type->traits.numeric.vector.component_count * (p_member_type->traits.numeric.scalar.width / SPIRV_BYTE_WIDTH);
        p_member_var->size = size;
      } break;

      case SpvOpTypeMatrix: {
        if (p_member_var->decoration_flags & SPV_REFLECT_DECORATION_COLUMN_MAJOR) {
          p_member_var->size = p_member_var->numeric.matrix.column_count * p_member_var->numeric.matrix.stride;
        } else if (p_member_var->decoration_flags & SPV_REFLECT_DECORATION_ROW_MAJOR) {
          p_member_var->size = p_member_var->numeric.matrix.row_count * p_member_var->numeric.matrix.stride;
        }
      } break;

      case SpvOpTypeArray: {
        // If array of structs, parse members first...
        bool is_struct = (p_member_type->type_flags & SPV_REFLECT_TYPE_FLAG_STRUCT) == SPV_REFLECT_TYPE_FLAG_STRUCT;
        if (is_struct) {
          if (p_member_var->flags & SPV_REFLECT_VARIABLE_FLAGS_PHYSICAL_POINTER_COPY) {
            p_member_var->size = GetPhysicalPointerStructSize(p_parser, p_member_type->id);
          } else {
            SpvReflectResult result =
                ParseDescriptorBlockVariableSizes(p_parser, p_module, false, true, is_parent_rta, p_member_var);
            if (result != SPV_REFLECT_RESULT_SUCCESS) {
              return result;
            }
          }
        }
        // ...then array
        uint32_t element_count = (p_member_var->array.dims_count > 0 ? 1 : 0);
        for (uint32_t i = 0; i < p_member_var->array.dims_count; ++i) {
          if (p_member_var->array.spec_constant_op_ids[i] != (uint32_t)INVALID_VALUE) {
            // Force size to be reported as 0 to effectively disable buffer size validation, since
            // the value is unreliable anyway as only valid for the default values of the SCs involved.
            element_count = 0;
          }
          element_count *= p_member_var->array.dims[i];
        }
        p_member_var->size = element_count * p_member_var->array.stride;
      } break;

      case SpvOpTypeRuntimeArray: {
        bool is_struct = (p_member_type->type_flags & SPV_REFLECT_TYPE_FLAG_STRUCT) == SPV_REFLECT_TYPE_FLAG_STRUCT;
        if (is_struct) {
          SpvReflectResult result = ParseDescriptorBlockVariableSizes(p_parser, p_module, false, true, true, p_member_var);
          if (result != SPV_REFLECT_RESULT_SUCCESS) {
            return result;
          }
        }
      } break;

      case SpvOpTypePointer: {
        // Reference. Get to underlying struct type.
        SpvReflectPrvNode* p_member_type_node = FindNode(p_parser, p_member_type->id);
        if (IsNull(p_member_type_node)) {
          return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
        }
        // Get the pointee type
        p_member_type = FindType(p_module, p_member_type_node->type_id);
        if (IsNull(p_member_type)) {
          return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
        }
        assert(p_member_type->op == SpvOpTypeStruct);
        FALLTHROUGH;
      }

      case SpvOpTypeStruct: {
        if (p_member_var->flags & SPV_REFLECT_VARIABLE_FLAGS_PHYSICAL_POINTER_COPY) {
          p_member_var->size = GetPhysicalPointerStructSize(p_parser, p_member_type->id);
        } else {
          SpvReflectResult result =
              ParseDescriptorBlockVariableSizes(p_parser, p_module, false, is_parent_aos, is_parent_rta, p_member_var);
          if (result != SPV_REFLECT_RESULT_SUCCESS) {
            return result;
          }
        }
      } break;

      default:
        break;
    }
  }

  // Structs can offset order don't need to match the index order, so first order by offset
  // example:
  //     OpMemberDecorate %struct 0 Offset 4
  //     OpMemberDecorate %struct 1 Offset 0
  SpvReflectBlockVariable** pp_member_offset_order =
      (SpvReflectBlockVariable**)calloc(p_var->member_count, sizeof(SpvReflectBlockVariable*));
  if (IsNull(pp_member_offset_order)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }

  uint32_t bottom_bound = 0;
  for (uint32_t i = 0; i < p_var->member_count; ++i) {
    uint32_t lowest_offset = UINT32_MAX;
    uint32_t member_index = 0;
    for (uint32_t j = 0; j < p_var->member_count; ++j) {
      const uint32_t offset = p_var->members[j].offset;
      if (offset < lowest_offset && offset >= bottom_bound) {
        member_index = j;
        lowest_offset = offset;
      }
    }
    pp_member_offset_order[i] = &p_var->members[member_index];
    bottom_bound = lowest_offset + 1;  // 2 index can't share the same offset
  }

  // Parse padded size using offset difference for all member except for the last entry...
  for (uint32_t i = 0; i < (p_var->member_count - 1); ++i) {
    SpvReflectBlockVariable* p_member_var = pp_member_offset_order[i];
    SpvReflectBlockVariable* p_next_member_var = pp_member_offset_order[i + 1];
    p_member_var->padded_size = p_next_member_var->offset - p_member_var->offset;
    if (p_member_var->size > p_member_var->padded_size) {
      p_member_var->size = p_member_var->padded_size;
    }
    if (is_parent_rta) {
      p_member_var->padded_size = p_member_var->size;
    }
  }

  // ...last entry just gets rounded up to near multiple of SPIRV_DATA_ALIGNMENT, which is 16 and
  // subtract the offset.
  // last entry == entry with largest offset value
  SpvReflectBlockVariable* p_last_member_var = pp_member_offset_order[p_var->member_count - 1];
  p_last_member_var->padded_size =
      RoundUp(p_last_member_var->offset + p_last_member_var->size, SPIRV_DATA_ALIGNMENT) - p_last_member_var->offset;
  if (p_last_member_var->size > p_last_member_var->padded_size) {
    p_last_member_var->size = p_last_member_var->padded_size;
  }
  if (is_parent_rta) {
    p_last_member_var->padded_size = p_last_member_var->size;
  }

  SafeFree(pp_member_offset_order);

  // If buffer ref, sizes are same as uint64_t
  if (is_parent_ref) {
    p_var->size = p_var->padded_size = 8;
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  // @TODO validate this with assertion
  p_var->size = p_last_member_var->offset + p_last_member_var->padded_size;
  p_var->padded_size = p_var->size;

  return SPV_REFLECT_RESULT_SUCCESS;
}

static void MarkSelfAndAllMemberVarsAsUsed(SpvReflectBlockVariable* p_var) {
  // Clear the current variable's UNUSED flag
  p_var->flags &= ~SPV_REFLECT_VARIABLE_FLAGS_UNUSED;

  SpvOp op_type = p_var->type_description->op;
  switch (op_type) {
    default:
      break;

    case SpvOpTypeArray: {
    } break;

    case SpvOpTypeStruct: {
      for (uint32_t i = 0; i < p_var->member_count; ++i) {
        SpvReflectBlockVariable* p_member_var = &p_var->members[i];
        MarkSelfAndAllMemberVarsAsUsed(p_member_var);
      }
    } break;
  }
}

static SpvReflectResult ParseDescriptorBlockVariableUsage(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module,
                                                          SpvReflectPrvAccessChain* p_access_chain, uint32_t index_index,
                                                          SpvOp override_op_type, SpvReflectBlockVariable* p_var) {
  // Clear the current variable's UNUSED flag
  p_var->flags &= ~SPV_REFLECT_VARIABLE_FLAGS_UNUSED;

  // Parsing arrays requires overriding the op type for
  // for the lowest dim's element type.
  SpvReflectTypeDescription* p_type = p_var->type_description;
  SpvOp op_type = p_type->op;
  if (override_op_type != (SpvOp)INVALID_VALUE) {
    op_type = override_op_type;
  }

  switch (op_type) {
    default:
      break;

    case SpvOpTypeArray: {
      // Parse through array's type hierarchy to find the actual/non-array element type
      while ((p_type->op == SpvOpTypeArray) && (index_index < p_access_chain->index_count)) {
        // Find the array element type id
        SpvReflectPrvNode* p_node = FindNode(p_parser, p_type->id);
        if (p_node == NULL) {
          return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
        }
        uint32_t element_type_id = p_node->array_traits.element_type_id;
        // Get the array element type
        p_type = FindType(p_module, element_type_id);
        if (p_type == NULL) {
          return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
        }
        // Next access chain index
        index_index += 1;
      }

      // Only continue parsing if there's remaining indices in the access
      // chain. If the end of the access chain has been reached then all
      // remaining variables (including those in struct hierarchies)
      // are considered USED.
      //
      // See: https://github.com/KhronosGroup/SPIRV-Reflect/issues/78
      //
      if (index_index < p_access_chain->index_count) {
        // Parse current var again with a type override and advanced index index
        SpvReflectResult result =
            ParseDescriptorBlockVariableUsage(p_parser, p_module, p_access_chain, index_index, p_type->op, p_var);
        if (result != SPV_REFLECT_RESULT_SUCCESS) {
          return result;
        }
      } else {
        // Clear UNUSED flag for remaining variables
        MarkSelfAndAllMemberVarsAsUsed(p_var);
      }
    } break;

    case SpvOpTypePointer: {
      // Reference. Get to underlying struct type.
      SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
      if (IsNull(p_type_node)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
      // Get the pointee type
      p_type = FindType(p_module, p_type_node->type_id);
      if (IsNull(p_type)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
      if (p_type->op != SpvOpTypeStruct) {
        break;
      }
      FALLTHROUGH;
    }

    case SpvOpTypeStruct: {
      assert(p_var->member_count > 0);
      if (p_var->member_count == 0) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_BLOCK_DATA;
      }

      // The access chain can have zero indexes, if used for a runtime array
      if (p_access_chain->index_count == 0) {
        return SPV_REFLECT_RESULT_SUCCESS;
      }

      // Get member variable at the access's chain current index
      uint32_t index = p_access_chain->indexes[index_index];
      if (index >= p_var->member_count) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_BLOCK_MEMBER_REFERENCE;
      }
      SpvReflectBlockVariable* p_member_var = &p_var->members[index];

      bool is_pointer_to_pointer = IsPointerToPointer(p_parser, p_access_chain->result_type_id);
      if (is_pointer_to_pointer) {
        // Remember block var for this access chain for downstream dereference
        p_access_chain->block_var = p_member_var;
      }

      // Next access chain index
      index_index += 1;

      // Only continue parsing if there's remaining indices in the access
      // chain. If the end of the access chain has been reach then all
      // remaining variables (including those in struct hierarchies)
      // are considered USED.
      //
      // See: https://github.com/KhronosGroup/SPIRV-Reflect/issues/78
      //
      if (index_index < p_access_chain->index_count) {
        SpvReflectResult result =
            ParseDescriptorBlockVariableUsage(p_parser, p_module, p_access_chain, index_index, (SpvOp)INVALID_VALUE, p_member_var);
        if (result != SPV_REFLECT_RESULT_SUCCESS) {
          return result;
        }
      } else if (!is_pointer_to_pointer) {
        // Clear UNUSED flag for remaining variables
        MarkSelfAndAllMemberVarsAsUsed(p_member_var);
      }
    } break;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseDescriptorBlocks(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  if (p_module->descriptor_binding_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  p_parser->physical_pointer_struct_count = 0;

  for (uint32_t descriptor_index = 0; descriptor_index < p_module->descriptor_binding_count; ++descriptor_index) {
    SpvReflectDescriptorBinding* p_descriptor = &(p_module->descriptor_bindings[descriptor_index]);
    SpvReflectTypeDescription* p_type = p_descriptor->type_description;
    if ((p_descriptor->descriptor_type != SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER) &&
        (p_descriptor->descriptor_type != SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER)) {
      continue;
    }

    // Mark UNUSED
    p_descriptor->block.flags |= SPV_REFLECT_VARIABLE_FLAGS_UNUSED;
    p_parser->physical_pointer_count = 0;
    // Parse descriptor block
    SpvReflectResult result = ParseDescriptorBlockVariable(p_parser, p_module, p_type, &p_descriptor->block);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }

    for (uint32_t access_chain_index = 0; access_chain_index < p_parser->access_chain_count; ++access_chain_index) {
      SpvReflectPrvAccessChain* p_access_chain = &(p_parser->access_chains[access_chain_index]);
      // Skip any access chains that aren't touching this descriptor block
      if (p_descriptor->spirv_id != p_access_chain->base_id) {
        continue;
      }
      result = ParseDescriptorBlockVariableUsage(p_parser, p_module, p_access_chain, 0, (SpvOp)INVALID_VALUE, &p_descriptor->block);
      if (result != SPV_REFLECT_RESULT_SUCCESS) {
        return result;
      }
    }

    p_descriptor->block.name = p_descriptor->name;

    bool is_parent_rta = (p_descriptor->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    result = ParseDescriptorBlockVariableSizes(p_parser, p_module, true, false, is_parent_rta, &p_descriptor->block);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }

    if (is_parent_rta) {
      p_descriptor->block.size = 0;
      p_descriptor->block.padded_size = 0;
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseFormat(const SpvReflectTypeDescription* p_type, SpvReflectFormat* p_format) {
  SpvReflectResult result = SPV_REFLECT_RESULT_ERROR_INTERNAL_ERROR;
  bool signedness = (p_type->traits.numeric.scalar.signedness != 0);
  uint32_t bit_width = p_type->traits.numeric.scalar.width;
  if (p_type->type_flags & SPV_REFLECT_TYPE_FLAG_VECTOR) {
    uint32_t component_count = p_type->traits.numeric.vector.component_count;
    if (p_type->type_flags & SPV_REFLECT_TYPE_FLAG_FLOAT) {
      switch (bit_width) {
        case 16: {
          switch (component_count) {
            case 2:
              *p_format = SPV_REFLECT_FORMAT_R16G16_SFLOAT;
              break;
            case 3:
              *p_format = SPV_REFLECT_FORMAT_R16G16B16_SFLOAT;
              break;
            case 4:
              *p_format = SPV_REFLECT_FORMAT_R16G16B16A16_SFLOAT;
              break;
          }
        } break;

        case 32: {
          switch (component_count) {
            case 2:
              *p_format = SPV_REFLECT_FORMAT_R32G32_SFLOAT;
              break;
            case 3:
              *p_format = SPV_REFLECT_FORMAT_R32G32B32_SFLOAT;
              break;
            case 4:
              *p_format = SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT;
              break;
          }
        } break;

        case 64: {
          switch (component_count) {
            case 2:
              *p_format = SPV_REFLECT_FORMAT_R64G64_SFLOAT;
              break;
            case 3:
              *p_format = SPV_REFLECT_FORMAT_R64G64B64_SFLOAT;
              break;
            case 4:
              *p_format = SPV_REFLECT_FORMAT_R64G64B64A64_SFLOAT;
              break;
          }
        }
      }
      result = SPV_REFLECT_RESULT_SUCCESS;
    } else if (p_type->type_flags & (SPV_REFLECT_TYPE_FLAG_INT | SPV_REFLECT_TYPE_FLAG_BOOL)) {
      switch (bit_width) {
        case 16: {
          switch (component_count) {
            case 2:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R16G16_SINT : SPV_REFLECT_FORMAT_R16G16_UINT;
              break;
            case 3:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R16G16B16_SINT : SPV_REFLECT_FORMAT_R16G16B16_UINT;
              break;
            case 4:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R16G16B16A16_SINT : SPV_REFLECT_FORMAT_R16G16B16A16_UINT;
              break;
          }
        } break;

        case 32: {
          switch (component_count) {
            case 2:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R32G32_SINT : SPV_REFLECT_FORMAT_R32G32_UINT;
              break;
            case 3:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R32G32B32_SINT : SPV_REFLECT_FORMAT_R32G32B32_UINT;
              break;
            case 4:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R32G32B32A32_SINT : SPV_REFLECT_FORMAT_R32G32B32A32_UINT;
              break;
          }
        } break;

        case 64: {
          switch (component_count) {
            case 2:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R64G64_SINT : SPV_REFLECT_FORMAT_R64G64_UINT;
              break;
            case 3:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R64G64B64_SINT : SPV_REFLECT_FORMAT_R64G64B64_UINT;
              break;
            case 4:
              *p_format = signedness ? SPV_REFLECT_FORMAT_R64G64B64A64_SINT : SPV_REFLECT_FORMAT_R64G64B64A64_UINT;
              break;
          }
        }
      }
      result = SPV_REFLECT_RESULT_SUCCESS;
    }
  } else if (p_type->type_flags & SPV_REFLECT_TYPE_FLAG_FLOAT) {
    switch (bit_width) {
      case 16:
        *p_format = SPV_REFLECT_FORMAT_R16_SFLOAT;
        break;
      case 32:
        *p_format = SPV_REFLECT_FORMAT_R32_SFLOAT;
        break;
      case 64:
        *p_format = SPV_REFLECT_FORMAT_R64_SFLOAT;
        break;
    }
    result = SPV_REFLECT_RESULT_SUCCESS;
  } else if (p_type->type_flags & (SPV_REFLECT_TYPE_FLAG_INT | SPV_REFLECT_TYPE_FLAG_BOOL)) {
    switch (bit_width) {
      case 16:
        *p_format = signedness ? SPV_REFLECT_FORMAT_R16_SINT : SPV_REFLECT_FORMAT_R16_UINT;
        break;
        break;
      case 32:
        *p_format = signedness ? SPV_REFLECT_FORMAT_R32_SINT : SPV_REFLECT_FORMAT_R32_UINT;
        break;
        break;
      case 64:
        *p_format = signedness ? SPV_REFLECT_FORMAT_R64_SINT : SPV_REFLECT_FORMAT_R64_UINT;
        break;
    }
    result = SPV_REFLECT_RESULT_SUCCESS;
  } else if (p_type->type_flags & SPV_REFLECT_TYPE_FLAG_STRUCT) {
    *p_format = SPV_REFLECT_FORMAT_UNDEFINED;
    result = SPV_REFLECT_RESULT_SUCCESS;
  }
  return result;
}

static SpvReflectResult ParseInterfaceVariable(SpvReflectPrvParser* p_parser,
                                               const SpvReflectPrvDecorations* p_var_node_decorations,
                                               const SpvReflectPrvDecorations* p_type_node_decorations,
                                               SpvReflectShaderModule* p_module, SpvReflectTypeDescription* p_type,
                                               SpvReflectInterfaceVariable* p_var, bool* p_has_built_in) {
  SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
  if (IsNull(p_type_node)) {
    return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
  }

  if (p_type->member_count > 0) {
    p_var->member_count = p_type->member_count;
    p_var->members = (SpvReflectInterfaceVariable*)calloc(p_var->member_count, sizeof(*p_var->members));
    if (IsNull(p_var->members)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }

    for (uint32_t member_index = 0; member_index < p_type_node->member_count; ++member_index) {
      SpvReflectPrvDecorations* p_member_decorations = &p_type_node->member_decorations[member_index];
      SpvReflectTypeDescription* p_member_type = &p_type->members[member_index];
      SpvReflectInterfaceVariable* p_member_var = &p_var->members[member_index];

      // Storage class is the same throughout the whole struct
      p_member_var->storage_class = p_var->storage_class;

      SpvReflectResult result =
          ParseInterfaceVariable(p_parser, NULL, p_member_decorations, p_module, p_member_type, p_member_var, p_has_built_in);
      if (result != SPV_REFLECT_RESULT_SUCCESS) {
        SPV_REFLECT_ASSERT(false);
        return result;
      }
    }
  }

  p_var->name = p_type_node->name;
  p_var->decoration_flags = ApplyDecorations(p_type_node_decorations);
  if (p_var_node_decorations != NULL) {
    p_var->decoration_flags |= ApplyDecorations(p_var_node_decorations);
  } else {
    // Apply member decoration values to struct members
    p_var->location = p_type_node_decorations->location.value;
    p_var->component = p_type_node_decorations->component.value;
  }

  p_var->built_in = p_type_node_decorations->built_in;
  ApplyNumericTraits(p_type, &p_var->numeric);
  if (p_type->op == SpvOpTypeArray) {
    ApplyArrayTraits(p_type, &p_var->array);
  }

  p_var->type_description = p_type;

  *p_has_built_in |= p_type_node_decorations->is_built_in;

  // Only parse format for interface variables that are input or output
  if ((p_var->storage_class == SpvStorageClassInput) || (p_var->storage_class == SpvStorageClassOutput)) {
    SpvReflectResult result = ParseFormat(p_var->type_description, &p_var->format);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      SPV_REFLECT_ASSERT(false);
      return result;
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseInterfaceVariables(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module,
                                                SpvReflectEntryPoint* p_entry, uint32_t interface_variable_count,
                                                uint32_t* p_interface_variable_ids) {
  if (interface_variable_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  p_entry->interface_variable_count = interface_variable_count;
  p_entry->input_variable_count = 0;
  p_entry->output_variable_count = 0;
  for (size_t i = 0; i < interface_variable_count; ++i) {
    uint32_t var_result_id = *(p_interface_variable_ids + i);
    SpvReflectPrvNode* p_node = FindNode(p_parser, var_result_id);
    if (IsNull(p_node)) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }

    if (p_node->storage_class == SpvStorageClassInput) {
      p_entry->input_variable_count += 1;
    } else if (p_node->storage_class == SpvStorageClassOutput) {
      p_entry->output_variable_count += 1;
    }
  }

  if (p_entry->input_variable_count > 0) {
    p_entry->input_variables =
        (SpvReflectInterfaceVariable**)calloc(p_entry->input_variable_count, sizeof(*(p_entry->input_variables)));
    if (IsNull(p_entry->input_variables)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }

  if (p_entry->output_variable_count > 0) {
    p_entry->output_variables =
        (SpvReflectInterfaceVariable**)calloc(p_entry->output_variable_count, sizeof(*(p_entry->output_variables)));
    if (IsNull(p_entry->output_variables)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }

  if (p_entry->interface_variable_count > 0) {
    p_entry->interface_variables =
        (SpvReflectInterfaceVariable*)calloc(p_entry->interface_variable_count, sizeof(*(p_entry->interface_variables)));
    if (IsNull(p_entry->interface_variables)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }

  size_t input_index = 0;
  size_t output_index = 0;
  for (size_t i = 0; i < interface_variable_count; ++i) {
    uint32_t var_result_id = *(p_interface_variable_ids + i);
    SpvReflectPrvNode* p_node = FindNode(p_parser, var_result_id);
    if (IsNull(p_node)) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }

    SpvReflectTypeDescription* p_type = FindType(p_module, p_node->type_id);
    if (IsNull(p_node) || IsNull(p_type)) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }
    // If the type is a pointer, resolve it
    if (p_type->op == SpvOpTypePointer) {
      // Find the type's node
      SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
      if (IsNull(p_type_node)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
      // Should be the resolved type
      p_type = FindType(p_module, p_type_node->type_id);
      if (IsNull(p_type)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
    }

    SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
    if (IsNull(p_type_node)) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }

    SpvReflectInterfaceVariable* p_var = &(p_entry->interface_variables[i]);
    p_var->storage_class = p_node->storage_class;

    bool has_built_in = p_node->decorations.is_built_in;
    SpvReflectResult result =
        ParseInterfaceVariable(p_parser, &p_node->decorations, &p_type_node->decorations, p_module, p_type, p_var, &has_built_in);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      SPV_REFLECT_ASSERT(false);
      return result;
    }

    // Input and output variables
    if (p_var->storage_class == SpvStorageClassInput) {
      p_entry->input_variables[input_index] = p_var;
      ++input_index;
    } else if (p_node->storage_class == SpvStorageClassOutput) {
      p_entry->output_variables[output_index] = p_var;
      ++output_index;
    }

    // SPIR-V result id
    p_var->spirv_id = p_node->result_id;
    // Name
    p_var->name = p_node->name;
    // Semantic
    p_var->semantic = p_node->decorations.semantic.value;

    // Decorate with built-in if any member is built-in
    if (has_built_in) {
      p_var->decoration_flags |= SPV_REFLECT_DECORATION_BUILT_IN;
    }

    // Location is decorated on OpVariable node, not the type node.
    p_var->location = p_node->decorations.location.value;
    p_var->component = p_node->decorations.component.value;
    p_var->word_offset.location = p_node->decorations.location.word_offset;

    // Built in
    if (p_node->decorations.is_built_in) {
      p_var->built_in = p_node->decorations.built_in;
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult EnumerateAllPushConstants(SpvReflectShaderModule* p_module, size_t* p_push_constant_count,
                                                  uint32_t** p_push_constants) {
  *p_push_constant_count = p_module->push_constant_block_count;
  if (*p_push_constant_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }
  *p_push_constants = (uint32_t*)calloc(*p_push_constant_count, sizeof(**p_push_constants));

  if (IsNull(*p_push_constants)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }

  for (size_t i = 0; i < *p_push_constant_count; ++i) {
    (*p_push_constants)[i] = p_module->push_constant_blocks[i].spirv_id;
  }
  qsort(*p_push_constants, *p_push_constant_count, sizeof(**p_push_constants), SortCompareUint32);
  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult TraverseCallGraph(SpvReflectPrvParser* p_parser, SpvReflectPrvFunction* p_func, size_t* p_func_count,
                                          uint32_t* p_func_ids, uint32_t depth) {
  if (depth > p_parser->function_count) {
    // Vulkan does not permit recursion (Vulkan spec Appendix A):
    //   "Recursion: The static function-call graph for an entry point must not
    //    contain cycles."
    return SPV_REFLECT_RESULT_ERROR_SPIRV_RECURSION;
  }
  if (IsNotNull(p_func_ids)) {
    p_func_ids[(*p_func_count)++] = p_func->id;
  } else {
    ++*p_func_count;
  }
  for (size_t i = 0; i < p_func->callee_count; ++i) {
    SpvReflectResult result = TraverseCallGraph(p_parser, p_func->callee_ptrs[i], p_func_count, p_func_ids, depth + 1);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }
  }
  return SPV_REFLECT_RESULT_SUCCESS;
}

static uint32_t GetUint32Constant(SpvReflectPrvParser* p_parser, uint32_t id) {
  uint32_t result = (uint32_t)INVALID_VALUE;
  SpvReflectPrvNode* p_node = FindNode(p_parser, id);
  if (p_node && p_node->op == SpvOpConstant) {
    UNCHECKED_READU32(p_parser, p_node->word_offset + 3, result);
  }
  return result;
}

static bool HasByteAddressBufferOffset(SpvReflectPrvNode* p_node, SpvReflectDescriptorBinding* p_binding) {
  return IsNotNull(p_node) && IsNotNull(p_binding) && p_node->op == SpvOpAccessChain && p_node->word_count == 6 &&
         (p_binding->user_type == SPV_REFLECT_USER_TYPE_BYTE_ADDRESS_BUFFER ||
          p_binding->user_type == SPV_REFLECT_USER_TYPE_RW_BYTE_ADDRESS_BUFFER);
}

static SpvReflectResult ParseByteAddressBuffer(SpvReflectPrvParser* p_parser, SpvReflectPrvNode* p_node,
                                               SpvReflectDescriptorBinding* p_binding) {
  const SpvReflectResult not_found = SPV_REFLECT_RESULT_SUCCESS;
  if (!HasByteAddressBufferOffset(p_node, p_binding)) {
    return not_found;
  }

  uint32_t offset = 0;  // starting offset

  uint32_t base_id = 0;
  // expect first index of 2D access is zero
  UNCHECKED_READU32(p_parser, p_node->word_offset + 4, base_id);
  if (GetUint32Constant(p_parser, base_id) != 0) {
    return not_found;
  }
  UNCHECKED_READU32(p_parser, p_node->word_offset + 5, base_id);
  SpvReflectPrvNode* p_next_node = FindNode(p_parser, base_id);
  if (IsNull(p_next_node)) {
    return not_found;
  } else if (p_next_node->op == SpvOpConstant) {
    // The access chain might just be a constant right to the offset
    offset = GetUint32Constant(p_parser, base_id);
    p_binding->byte_address_buffer_offsets[p_binding->byte_address_buffer_offset_count] = offset;
    p_binding->byte_address_buffer_offset_count++;
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  // there is usually 2 (sometimes 3) instrucitons that make up the arithmetic logic to calculate the offset
  SpvReflectPrvNode* arithmetic_node_stack[8];
  uint32_t arithmetic_count = 0;

  while (IsNotNull(p_next_node)) {
    if (p_next_node->op == SpvOpLoad || p_next_node->op == SpvOpBitcast || p_next_node->op == SpvOpConstant) {
      break;  // arithmetic starts here
    }
    arithmetic_node_stack[arithmetic_count++] = p_next_node;
    if (arithmetic_count >= 8) {
      return not_found;
    }

    UNCHECKED_READU32(p_parser, p_next_node->word_offset + 3, base_id);
    p_next_node = FindNode(p_parser, base_id);
  }

  const uint32_t count = arithmetic_count;
  for (uint32_t i = 0; i < count; i++) {
    p_next_node = arithmetic_node_stack[--arithmetic_count];
    // All arithmetic ops takes 2 operands, assumption is the 2nd operand has the constant
    UNCHECKED_READU32(p_parser, p_next_node->word_offset + 4, base_id);
    uint32_t value = GetUint32Constant(p_parser, base_id);
    if (value == INVALID_VALUE) {
      return not_found;
    }

    switch (p_next_node->op) {
      case SpvOpShiftRightLogical:
        offset >>= value;
        break;
      case SpvOpIAdd:
        offset += value;
        break;
      case SpvOpISub:
        offset -= value;
        break;
      case SpvOpIMul:
        offset *= value;
        break;
      case SpvOpUDiv:
        offset /= value;
        break;
      case SpvOpSDiv:
        // OpConstant might be signed, but value should never be negative
        assert((int32_t)value > 0);
        offset /= value;
        break;
      default:
        return not_found;
    }
  }

  p_binding->byte_address_buffer_offsets[p_binding->byte_address_buffer_offset_count] = offset;
  p_binding->byte_address_buffer_offset_count++;
  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseStaticallyUsedResources(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module,
                                                     SpvReflectEntryPoint* p_entry, size_t uniform_count, uint32_t* uniforms,
                                                     size_t push_constant_count, uint32_t* push_constants) {
  // Find function with the right id
  SpvReflectPrvFunction* p_func = NULL;
  for (size_t i = 0; i < p_parser->function_count; ++i) {
    if (p_parser->functions[i].id == p_entry->id) {
      p_func = &(p_parser->functions[i]);
      break;
    }
  }
  if (p_func == NULL) {
    return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
  }

  size_t called_function_count = 0;
  SpvReflectResult result = TraverseCallGraph(p_parser, p_func, &called_function_count, NULL, 0);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    return result;
  }

  uint32_t* p_called_functions = NULL;
  if (called_function_count > 0) {
    p_called_functions = (uint32_t*)calloc(called_function_count, sizeof(*p_called_functions));
    if (IsNull(p_called_functions)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }

  called_function_count = 0;
  result = TraverseCallGraph(p_parser, p_func, &called_function_count, p_called_functions, 0);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    SafeFree(p_called_functions);
    return result;
  }

  if (called_function_count > 0) {
    qsort(p_called_functions, called_function_count, sizeof(*p_called_functions), SortCompareUint32);
  }
  called_function_count = DedupSortedUint32(p_called_functions, called_function_count);

  uint32_t used_acessed_count = 0;
  for (size_t i = 0, j = 0; i < called_function_count; ++i) {
    // No need to bounds check j because a missing ID issue would have been
    // found during TraverseCallGraph
    while (p_parser->functions[j].id != p_called_functions[i]) {
      ++j;
    }
    used_acessed_count += p_parser->functions[j].accessed_variable_count;
  }
  SpvReflectPrvAccessedVariable* p_used_accesses = NULL;
  if (used_acessed_count > 0) {
    p_used_accesses = (SpvReflectPrvAccessedVariable*)calloc(used_acessed_count, sizeof(SpvReflectPrvAccessedVariable));
    if (IsNull(p_used_accesses)) {
      SafeFree(p_called_functions);
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
  }
  used_acessed_count = 0;
  for (size_t i = 0, j = 0; i < called_function_count; ++i) {
    while (p_parser->functions[j].id != p_called_functions[i]) {
      ++j;
    }

    memcpy(&p_used_accesses[used_acessed_count], p_parser->functions[j].accessed_variables,
           p_parser->functions[j].accessed_variable_count * sizeof(SpvReflectPrvAccessedVariable));
    used_acessed_count += p_parser->functions[j].accessed_variable_count;
  }
  SafeFree(p_called_functions);

  if (used_acessed_count > 0) {
    qsort(p_used_accesses, used_acessed_count, sizeof(*p_used_accesses), SortCompareAccessedVariable);
  }

  // Do set intersection to find the used uniform and push constants
  size_t used_uniform_count = 0;
  result = IntersectSortedAccessedVariable(p_used_accesses, used_acessed_count, uniforms, uniform_count, &p_entry->used_uniforms,
                                           &used_uniform_count);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    SafeFree(p_used_accesses);
    return result;
  }

  size_t used_push_constant_count = 0;
  result = IntersectSortedAccessedVariable(p_used_accesses, used_acessed_count, push_constants, push_constant_count,
                                           &p_entry->used_push_constants, &used_push_constant_count);
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    SafeFree(p_used_accesses);
    return result;
  }

  for (uint32_t i = 0; i < p_module->descriptor_binding_count; ++i) {
    SpvReflectDescriptorBinding* p_binding = &p_module->descriptor_bindings[i];
    uint32_t byte_address_buffer_offset_count = 0;

    for (uint32_t j = 0; j < used_acessed_count; j++) {
      if (p_used_accesses[j].variable_ptr == p_binding->spirv_id) {
        p_binding->accessed = 1;

        if (HasByteAddressBufferOffset(p_used_accesses[j].p_node, p_binding)) {
          byte_address_buffer_offset_count++;
        }
      }
    }

    // only if SPIR-V has ByteAddressBuffer user type
    if (byte_address_buffer_offset_count > 0) {
      bool multi_entrypoint = p_binding->byte_address_buffer_offset_count > 0;
      if (multi_entrypoint) {
        // If there is a 2nd entrypoint, we can have multiple entry points, in this case we want to just combine the accessed
        // offsets and then de-duplicate it
        uint32_t* prev_byte_address_buffer_offsets = p_binding->byte_address_buffer_offsets;
        p_binding->byte_address_buffer_offsets =
            (uint32_t*)calloc(byte_address_buffer_offset_count + p_binding->byte_address_buffer_offset_count, sizeof(uint32_t));
        memcpy(p_binding->byte_address_buffer_offsets, prev_byte_address_buffer_offsets,
               sizeof(uint32_t) * p_binding->byte_address_buffer_offset_count);
        SafeFree(prev_byte_address_buffer_offsets);
      } else {
        // possible not all allocated offset slots are used, but this will be a max per binding
        p_binding->byte_address_buffer_offsets = (uint32_t*)calloc(byte_address_buffer_offset_count, sizeof(uint32_t));
      }

      if (IsNull(p_binding->byte_address_buffer_offsets)) {
        SafeFree(p_used_accesses);
        return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
      }

      for (uint32_t j = 0; j < used_acessed_count; j++) {
        if (p_used_accesses[j].variable_ptr == p_binding->spirv_id) {
          result = ParseByteAddressBuffer(p_parser, p_used_accesses[j].p_node, p_binding);
          if (result != SPV_REFLECT_RESULT_SUCCESS) {
            SafeFree(p_used_accesses);
            return result;
          }
        }
      }

      if (multi_entrypoint) {
        qsort(p_binding->byte_address_buffer_offsets, p_binding->byte_address_buffer_offset_count,
              sizeof(*(p_binding->byte_address_buffer_offsets)), SortCompareUint32);
        p_binding->byte_address_buffer_offset_count =
            (uint32_t)DedupSortedUint32(p_binding->byte_address_buffer_offsets, p_binding->byte_address_buffer_offset_count);
      }
    }
  }

  SafeFree(p_used_accesses);

  p_entry->used_uniform_count = (uint32_t)used_uniform_count;
  p_entry->used_push_constant_count = (uint32_t)used_push_constant_count;

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseEntryPoints(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  if (p_parser->entry_point_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  p_module->entry_point_count = p_parser->entry_point_count;
  p_module->entry_points = (SpvReflectEntryPoint*)calloc(p_module->entry_point_count, sizeof(*(p_module->entry_points)));
  if (IsNull(p_module->entry_points)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }

  SpvReflectResult result;
  size_t uniform_count = 0;
  uint32_t* uniforms = NULL;
  if ((result = EnumerateAllUniforms(p_module, &uniform_count, &uniforms)) != SPV_REFLECT_RESULT_SUCCESS) {
    return result;
  }
  size_t push_constant_count = 0;
  uint32_t* push_constants = NULL;
  if ((result = EnumerateAllPushConstants(p_module, &push_constant_count, &push_constants)) != SPV_REFLECT_RESULT_SUCCESS) {
    return result;
  }

  size_t entry_point_index = 0;
  for (size_t i = 0; entry_point_index < p_parser->entry_point_count && i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if (p_node->op != SpvOpEntryPoint) {
      continue;
    }

    SpvReflectEntryPoint* p_entry_point = &(p_module->entry_points[entry_point_index]);
    CHECKED_READU32_CAST(p_parser, p_node->word_offset + 1, SpvExecutionModel, p_entry_point->spirv_execution_model);
    CHECKED_READU32(p_parser, p_node->word_offset + 2, p_entry_point->id);

    switch (p_entry_point->spirv_execution_model) {
      default:
        break;
      case SpvExecutionModelVertex:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_VERTEX_BIT;
        break;
      case SpvExecutionModelTessellationControl:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        break;
      case SpvExecutionModelTessellationEvaluation:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        break;
      case SpvExecutionModelGeometry:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_GEOMETRY_BIT;
        break;
      case SpvExecutionModelFragment:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT;
        break;
      case SpvExecutionModelGLCompute:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT;
        break;
      case SpvExecutionModelTaskNV:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_TASK_BIT_NV;
        break;
      case SpvExecutionModelTaskEXT:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_TASK_BIT_EXT;
        break;
      case SpvExecutionModelMeshNV:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_MESH_BIT_NV;
        break;
      case SpvExecutionModelMeshEXT:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_MESH_BIT_EXT;
        break;
      case SpvExecutionModelRayGenerationKHR:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_RAYGEN_BIT_KHR;
        break;
      case SpvExecutionModelIntersectionKHR:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_INTERSECTION_BIT_KHR;
        break;
      case SpvExecutionModelAnyHitKHR:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_ANY_HIT_BIT_KHR;
        break;
      case SpvExecutionModelClosestHitKHR:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        break;
      case SpvExecutionModelMissKHR:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_MISS_BIT_KHR;
        break;
      case SpvExecutionModelCallableKHR:
        p_entry_point->shader_stage = SPV_REFLECT_SHADER_STAGE_CALLABLE_BIT_KHR;
        break;
    }

    ++entry_point_index;

    // Name length is required to calculate next operand
    uint32_t name_start_word_offset = 3;
    uint32_t name_length_with_terminator = 0;
    result =
        ReadStr(p_parser, p_node->word_offset + name_start_word_offset, 0, p_node->word_count, &name_length_with_terminator, NULL);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }
    p_entry_point->name = (const char*)(p_parser->spirv_code + p_node->word_offset + name_start_word_offset);

    uint32_t name_word_count = RoundUp(name_length_with_terminator, SPIRV_WORD_SIZE) / SPIRV_WORD_SIZE;
    uint32_t interface_variable_count = (p_node->word_count - (name_start_word_offset + name_word_count));
    uint32_t* p_interface_variables = NULL;
    if (interface_variable_count > 0) {
      p_interface_variables = (uint32_t*)calloc(interface_variable_count, sizeof(*(p_interface_variables)));
      if (IsNull(p_interface_variables)) {
        return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
      }
    }

    for (uint32_t var_index = 0; var_index < interface_variable_count; ++var_index) {
      uint32_t var_result_id = (uint32_t)INVALID_VALUE;
      uint32_t offset = name_start_word_offset + name_word_count + var_index;
      CHECKED_READU32(p_parser, p_node->word_offset + offset, var_result_id);
      p_interface_variables[var_index] = var_result_id;
    }

    result = ParseInterfaceVariables(p_parser, p_module, p_entry_point, interface_variable_count, p_interface_variables);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }
    SafeFree(p_interface_variables);

    result = ParseStaticallyUsedResources(p_parser, p_module, p_entry_point, uniform_count, uniforms, push_constant_count,
                                          push_constants);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }
  }

  SafeFree(uniforms);
  SafeFree(push_constants);

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseExecutionModes(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  assert(IsNotNull(p_parser));
  assert(IsNotNull(p_parser->nodes));
  assert(IsNotNull(p_module));

  if (IsNotNull(p_parser) && IsNotNull(p_parser->spirv_code) && IsNotNull(p_parser->nodes)) {
    for (size_t node_idx = 0; node_idx < p_parser->node_count; ++node_idx) {
      SpvReflectPrvNode* p_node = &(p_parser->nodes[node_idx]);
      if (p_node->op != SpvOpExecutionMode && p_node->op != SpvOpExecutionModeId) {
        continue;
      }

      // Read entry point id
      uint32_t entry_point_id = 0;
      CHECKED_READU32(p_parser, p_node->word_offset + 1, entry_point_id);

      // Find entry point
      SpvReflectEntryPoint* p_entry_point = NULL;
      for (size_t entry_point_idx = 0; entry_point_idx < p_module->entry_point_count; ++entry_point_idx) {
        if (p_module->entry_points[entry_point_idx].id == entry_point_id) {
          p_entry_point = &p_module->entry_points[entry_point_idx];
          break;
        }
      }
      // Bail if entry point is null
      if (IsNull(p_entry_point)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ENTRY_POINT;
      }

      // Read execution mode
      uint32_t execution_mode = (uint32_t)INVALID_VALUE;
      CHECKED_READU32(p_parser, p_node->word_offset + 2, execution_mode);

      // Parse execution mode
      switch (execution_mode) {
        case SpvExecutionModeInvocations: {
          CHECKED_READU32(p_parser, p_node->word_offset + 3, p_entry_point->invocations);
        } break;

        case SpvExecutionModeLocalSize: {
          CHECKED_READU32(p_parser, p_node->word_offset + 3, p_entry_point->local_size.x);
          CHECKED_READU32(p_parser, p_node->word_offset + 4, p_entry_point->local_size.y);
          CHECKED_READU32(p_parser, p_node->word_offset + 5, p_entry_point->local_size.z);
        } break;
        case SpvExecutionModeLocalSizeId: {
          uint32_t local_size_x_id = 0;
          uint32_t local_size_y_id = 0;
          uint32_t local_size_z_id = 0;
          CHECKED_READU32(p_parser, p_node->word_offset + 3, local_size_x_id);
          CHECKED_READU32(p_parser, p_node->word_offset + 4, local_size_y_id);
          CHECKED_READU32(p_parser, p_node->word_offset + 5, local_size_z_id);

          SpvReflectPrvNode* x_node = FindNode(p_parser, local_size_x_id);
          SpvReflectPrvNode* y_node = FindNode(p_parser, local_size_y_id);
          SpvReflectPrvNode* z_node = FindNode(p_parser, local_size_z_id);
          if (IsNotNull(x_node) && IsNotNull(y_node) && IsNotNull(z_node)) {
            if (IsSpecConstant(x_node)) {
              p_entry_point->local_size.x = (uint32_t)SPV_REFLECT_EXECUTION_MODE_SPEC_CONSTANT;
            } else {
              CHECKED_READU32(p_parser, x_node->word_offset + 3, p_entry_point->local_size.x);
            }

            if (IsSpecConstant(y_node)) {
              p_entry_point->local_size.y = (uint32_t)SPV_REFLECT_EXECUTION_MODE_SPEC_CONSTANT;
            } else {
              CHECKED_READU32(p_parser, y_node->word_offset + 3, p_entry_point->local_size.y);
            }

            if (IsSpecConstant(z_node)) {
              p_entry_point->local_size.z = (uint32_t)SPV_REFLECT_EXECUTION_MODE_SPEC_CONSTANT;
            } else {
              CHECKED_READU32(p_parser, z_node->word_offset + 3, p_entry_point->local_size.z);
            }
          }
        } break;

        case SpvExecutionModeOutputVertices: {
          CHECKED_READU32(p_parser, p_node->word_offset + 3, p_entry_point->output_vertices);
        } break;

        default:
          break;
      }
      p_entry_point->execution_mode_count++;
    }
    uint32_t* indices = (uint32_t*)calloc(p_module->entry_point_count, sizeof(indices));
    if (IsNull(indices)) {
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
    for (size_t entry_point_idx = 0; entry_point_idx < p_module->entry_point_count; ++entry_point_idx) {
      SpvReflectEntryPoint* p_entry_point = &p_module->entry_points[entry_point_idx];
      if (p_entry_point->execution_mode_count > 0) {
        p_entry_point->execution_modes =
            (SpvExecutionMode*)calloc(p_entry_point->execution_mode_count, sizeof(*p_entry_point->execution_modes));
        if (IsNull(p_entry_point->execution_modes)) {
          SafeFree(indices);
          return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
        }
      }
    }

    for (size_t node_idx = 0; node_idx < p_parser->node_count; ++node_idx) {
      SpvReflectPrvNode* p_node = &(p_parser->nodes[node_idx]);
      if (p_node->op != SpvOpExecutionMode) {
        continue;
      }

      // Read entry point id
      uint32_t entry_point_id = 0;
      CHECKED_READU32(p_parser, p_node->word_offset + 1, entry_point_id);

      // Find entry point
      SpvReflectEntryPoint* p_entry_point = NULL;
      uint32_t* idx = NULL;
      for (size_t entry_point_idx = 0; entry_point_idx < p_module->entry_point_count; ++entry_point_idx) {
        if (p_module->entry_points[entry_point_idx].id == entry_point_id) {
          p_entry_point = &p_module->entry_points[entry_point_idx];
          idx = &indices[entry_point_idx];
          break;
        }
      }

      // Read execution mode
      uint32_t execution_mode = (uint32_t)INVALID_VALUE;
      CHECKED_READU32(p_parser, p_node->word_offset + 2, execution_mode);
      p_entry_point->execution_modes[(*idx)++] = (SpvExecutionMode)execution_mode;
    }
    SafeFree(indices);
  }
  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParsePushConstantBlocks(SpvReflectPrvParser* p_parser, SpvReflectShaderModule* p_module) {
  for (size_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if ((p_node->op != SpvOpVariable) || (p_node->storage_class != SpvStorageClassPushConstant)) {
      continue;
    }

    p_module->push_constant_block_count += 1;
  }

  if (p_module->push_constant_block_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  p_module->push_constant_blocks =
      (SpvReflectBlockVariable*)calloc(p_module->push_constant_block_count, sizeof(*p_module->push_constant_blocks));
  if (IsNull(p_module->push_constant_blocks)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }

  p_parser->physical_pointer_struct_count = 0;
  uint32_t push_constant_index = 0;
  for (size_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_node = &(p_parser->nodes[i]);
    if ((p_node->op != SpvOpVariable) || (p_node->storage_class != SpvStorageClassPushConstant)) {
      continue;
    }

    SpvReflectTypeDescription* p_type = FindType(p_module, p_node->type_id);
    if (IsNull(p_node) || IsNull(p_type)) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }
    // If the type is a pointer, resolve it
    if (p_type->op == SpvOpTypePointer) {
      // Find the type's node
      SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
      if (IsNull(p_type_node)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
      // Should be the resolved type
      p_type = FindType(p_module, p_type_node->type_id);
      if (IsNull(p_type)) {
        return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
      }
    }

    SpvReflectPrvNode* p_type_node = FindNode(p_parser, p_type->id);
    if (IsNull(p_type_node)) {
      return SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE;
    }

    SpvReflectBlockVariable* p_push_constant = &p_module->push_constant_blocks[push_constant_index];
    p_push_constant->spirv_id = p_node->result_id;
    p_parser->physical_pointer_count = 0;
    SpvReflectResult result = ParseDescriptorBlockVariable(p_parser, p_module, p_type, p_push_constant);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }

    for (uint32_t access_chain_index = 0; access_chain_index < p_parser->access_chain_count; ++access_chain_index) {
      SpvReflectPrvAccessChain* p_access_chain = &(p_parser->access_chains[access_chain_index]);
      // Skip any access chains that aren't touching this push constant block
      if (p_push_constant->spirv_id != FindAccessChainBaseVariable(p_parser, p_access_chain)) {
        continue;
      }
      SpvReflectBlockVariable* p_var =
          (p_access_chain->base_id == p_push_constant->spirv_id) ? p_push_constant : GetRefBlkVar(p_parser, p_access_chain);
      result = ParseDescriptorBlockVariableUsage(p_parser, p_module, p_access_chain, 0, (SpvOp)INVALID_VALUE, p_var);
      if (result != SPV_REFLECT_RESULT_SUCCESS) {
        return result;
      }
    }

    p_push_constant->name = p_node->name;
    result = ParseDescriptorBlockVariableSizes(p_parser, p_module, true, false, false, p_push_constant);
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
      return result;
    }

    // Get minimum offset for whole Push Constant block
    // It is not valid SPIR-V to have an empty Push Constant Block
    p_push_constant->offset = UINT32_MAX;
    for (uint32_t k = 0; k < p_push_constant->member_count; ++k) {
      const uint32_t member_offset = p_push_constant->members[k].offset;
      p_push_constant->offset = Min(p_push_constant->offset, member_offset);
    }

    ++push_constant_index;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static int SortCompareDescriptorSet(const void* a, const void* b) {
  const SpvReflectDescriptorSet* p_elem_a = (const SpvReflectDescriptorSet*)a;
  const SpvReflectDescriptorSet* p_elem_b = (const SpvReflectDescriptorSet*)b;
  int value = (int)(p_elem_a->set) - (int)(p_elem_b->set);
  // We should never see duplicate descriptor set numbers in a shader; if so, a tiebreaker
  // would be needed here.
  assert(value != 0);
  return value;
}

static SpvReflectResult ParseEntrypointDescriptorSets(SpvReflectShaderModule* p_module) {
  // Update the entry point's sets
  for (uint32_t i = 0; i < p_module->entry_point_count; ++i) {
    SpvReflectEntryPoint* p_entry = &p_module->entry_points[i];
    for (uint32_t j = 0; j < p_entry->descriptor_set_count; ++j) {
      SafeFree(p_entry->descriptor_sets[j].bindings);
    }
    SafeFree(p_entry->descriptor_sets);
    p_entry->descriptor_set_count = 0;
    for (uint32_t j = 0; j < p_module->descriptor_set_count; ++j) {
      const SpvReflectDescriptorSet* p_set = &p_module->descriptor_sets[j];
      for (uint32_t k = 0; k < p_set->binding_count; ++k) {
        bool found = SearchSortedUint32(p_entry->used_uniforms, p_entry->used_uniform_count, p_set->bindings[k]->spirv_id);
        if (found) {
          ++p_entry->descriptor_set_count;
          break;
        }
      }
    }

    p_entry->descriptor_sets = NULL;
    if (p_entry->descriptor_set_count > 0) {
      p_entry->descriptor_sets = (SpvReflectDescriptorSet*)calloc(p_entry->descriptor_set_count, sizeof(*p_entry->descriptor_sets));
      if (IsNull(p_entry->descriptor_sets)) {
        return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
      }
    }
    p_entry->descriptor_set_count = 0;
    for (uint32_t j = 0; j < p_module->descriptor_set_count; ++j) {
      const SpvReflectDescriptorSet* p_set = &p_module->descriptor_sets[j];
      uint32_t count = 0;
      for (uint32_t k = 0; k < p_set->binding_count; ++k) {
        bool found = SearchSortedUint32(p_entry->used_uniforms, p_entry->used_uniform_count, p_set->bindings[k]->spirv_id);
        if (found) {
          ++count;
        }
      }
      if (count == 0) {
        continue;
      }
      SpvReflectDescriptorSet* p_entry_set = &p_entry->descriptor_sets[p_entry->descriptor_set_count++];
      p_entry_set->set = p_set->set;
      p_entry_set->bindings = (SpvReflectDescriptorBinding**)calloc(count, sizeof(*p_entry_set->bindings));
      if (IsNull(p_entry_set->bindings)) {
        return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
      }
      for (uint32_t k = 0; k < p_set->binding_count; ++k) {
        bool found = SearchSortedUint32(p_entry->used_uniforms, p_entry->used_uniform_count, p_set->bindings[k]->spirv_id);
        if (found) {
          p_entry_set->bindings[p_entry_set->binding_count++] = p_set->bindings[k];
        }
      }
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult ParseDescriptorSets(SpvReflectShaderModule* p_module) {
  // Count the descriptors in each set
  for (uint32_t i = 0; i < p_module->descriptor_binding_count; ++i) {
    SpvReflectDescriptorBinding* p_descriptor = &(p_module->descriptor_bindings[i]);

    // Look for a target set using the descriptor's set number
    SpvReflectDescriptorSet* p_target_set = NULL;
    for (uint32_t j = 0; j < SPV_REFLECT_MAX_DESCRIPTOR_SETS; ++j) {
      SpvReflectDescriptorSet* p_set = &p_module->descriptor_sets[j];
      if (p_set->set == p_descriptor->set) {
        p_target_set = p_set;
        break;
      }
    }

    // If a target set isn't found, find the first available one.
    if (IsNull(p_target_set)) {
      for (uint32_t j = 0; j < SPV_REFLECT_MAX_DESCRIPTOR_SETS; ++j) {
        SpvReflectDescriptorSet* p_set = &p_module->descriptor_sets[j];
        if (p_set->set == (uint32_t)INVALID_VALUE) {
          p_target_set = p_set;
          p_target_set->set = p_descriptor->set;
          break;
        }
      }
    }

    if (IsNull(p_target_set)) {
      return SPV_REFLECT_RESULT_ERROR_INTERNAL_ERROR;
    }

    p_target_set->binding_count += 1;
  }

  // Count the descriptor sets
  for (uint32_t i = 0; i < SPV_REFLECT_MAX_DESCRIPTOR_SETS; ++i) {
    const SpvReflectDescriptorSet* p_set = &p_module->descriptor_sets[i];
    if (p_set->set != (uint32_t)INVALID_VALUE) {
      p_module->descriptor_set_count += 1;
    }
  }

  // Sort the descriptor sets based on numbers
  if (p_module->descriptor_set_count > 0) {
    qsort(p_module->descriptor_sets, p_module->descriptor_set_count, sizeof(*(p_module->descriptor_sets)),
          SortCompareDescriptorSet);
  }

  // Build descriptor pointer array
  for (uint32_t i = 0; i < p_module->descriptor_set_count; ++i) {
    SpvReflectDescriptorSet* p_set = &(p_module->descriptor_sets[i]);
    p_set->bindings = (SpvReflectDescriptorBinding**)calloc(p_set->binding_count, sizeof(*(p_set->bindings)));

    uint32_t descriptor_index = 0;
    for (uint32_t j = 0; j < p_module->descriptor_binding_count; ++j) {
      SpvReflectDescriptorBinding* p_descriptor = &(p_module->descriptor_bindings[j]);
      if (p_descriptor->set == p_set->set) {
        assert(descriptor_index < p_set->binding_count);
        p_set->bindings[descriptor_index] = p_descriptor;
        ++descriptor_index;
      }
    }
  }

  return ParseEntrypointDescriptorSets(p_module);
}

static SpvReflectResult DisambiguateStorageBufferSrvUav(SpvReflectShaderModule* p_module) {
  if (p_module->descriptor_binding_count == 0) {
    return SPV_REFLECT_RESULT_SUCCESS;
  }

  for (uint32_t descriptor_index = 0; descriptor_index < p_module->descriptor_binding_count; ++descriptor_index) {
    SpvReflectDescriptorBinding* p_descriptor = &(p_module->descriptor_bindings[descriptor_index]);
    // Skip everything that isn't a STORAGE_BUFFER descriptor
    if (p_descriptor->descriptor_type != SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      continue;
    }

    //
    // Vulkan doesn't disambiguate between SRVs and UAVs so they
    // come back as STORAGE_BUFFER. The block parsing process will
    // mark a block as non-writable should any member of the block
    // or its descendants are non-writable.
    //
    if (p_descriptor->block.decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE) {
      p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_SRV;
    }
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

static SpvReflectResult SynchronizeDescriptorSets(SpvReflectShaderModule* p_module) {
  // Free and reset all descriptor set numbers
  for (uint32_t i = 0; i < SPV_REFLECT_MAX_DESCRIPTOR_SETS; ++i) {
    SpvReflectDescriptorSet* p_set = &p_module->descriptor_sets[i];
    SafeFree(p_set->bindings);
    p_set->binding_count = 0;
    p_set->set = (uint32_t)INVALID_VALUE;
  }
  // Set descriptor set count to zero
  p_module->descriptor_set_count = 0;

  SpvReflectResult result = ParseDescriptorSets(p_module);
  return result;
}

static SpvReflectResult CreateShaderModule(uint32_t flags, size_t size, const void* p_code, SpvReflectShaderModule* p_module) {
  // Initialize all module fields to zero
  memset(p_module, 0, sizeof(*p_module));

  // Allocate module internals
#ifdef __cplusplus
  p_module->_internal = (SpvReflectShaderModule::Internal*)calloc(1, sizeof(*(p_module->_internal)));
#else
  p_module->_internal = calloc(1, sizeof(*(p_module->_internal)));
#endif
  if (IsNull(p_module->_internal)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }
  // Copy flags
  p_module->_internal->module_flags = flags;
  // Figure out if we need to copy the SPIR-V code or not
  if (flags & SPV_REFLECT_MODULE_FLAG_NO_COPY) {
    // Set internal size and pointer to args passed in
    p_module->_internal->spirv_size = size;
#if defined(__cplusplus)
    p_module->_internal->spirv_code = const_cast<uint32_t*>(static_cast<const uint32_t*>(p_code));  // cast that const away
#else
    p_module->_internal->spirv_code = (void*)p_code;  // cast that const away
#endif
    p_module->_internal->spirv_word_count = (uint32_t)(size / SPIRV_WORD_SIZE);
  } else {
    // Allocate SPIR-V code storage
    p_module->_internal->spirv_size = size;
    p_module->_internal->spirv_code = (uint32_t*)calloc(1, p_module->_internal->spirv_size);
    p_module->_internal->spirv_word_count = (uint32_t)(size / SPIRV_WORD_SIZE);
    if (IsNull(p_module->_internal->spirv_code)) {
      SafeFree(p_module->_internal);
      return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
    }
    // Copy SPIR-V to code storage
    memcpy(p_module->_internal->spirv_code, p_code, size);
  }

  // Initialize everything to zero
  SpvReflectPrvParser parser;
  memset(&parser, 0, sizeof(SpvReflectPrvParser));

  // Create parser
  SpvReflectResult result = CreateParser(p_module->_internal->spirv_size, p_module->_internal->spirv_code, &parser);

  // Generator
  {
    const uint32_t* p_ptr = (const uint32_t*)p_module->_internal->spirv_code;
    p_module->generator = (SpvReflectGenerator)((*(p_ptr + 2) & 0xFFFF0000) >> 16);
  }

  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseNodes(&parser);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseStrings(&parser);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseSource(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseFunctions(&parser);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseMemberCounts(&parser);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseNames(&parser);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseDecorations(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }

  // Start of reflection data parsing
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    p_module->source_language = parser.source_language;
    p_module->source_language_version = parser.source_language_version;

    // Zero out descriptor set data
    p_module->descriptor_set_count = 0;
    memset(p_module->descriptor_sets, 0, SPV_REFLECT_MAX_DESCRIPTOR_SETS * sizeof(*p_module->descriptor_sets));
    // Initialize descriptor set numbers
    for (uint32_t set_number = 0; set_number < SPV_REFLECT_MAX_DESCRIPTOR_SETS; ++set_number) {
      p_module->descriptor_sets[set_number].set = (uint32_t)INVALID_VALUE;
    }
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseTypes(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseDescriptorBindings(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseDescriptorType(p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseUAVCounterBindings(p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseDescriptorBlocks(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParsePushConstantBlocks(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseEntryPoints(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseCapabilities(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS && p_module->entry_point_count > 0) {
    SpvReflectEntryPoint* p_entry = &(p_module->entry_points[0]);
    p_module->entry_point_name = p_entry->name;
    p_module->entry_point_id = p_entry->id;
    p_module->spirv_execution_model = p_entry->spirv_execution_model;
    p_module->shader_stage = p_entry->shader_stage;
    p_module->input_variable_count = p_entry->input_variable_count;
    p_module->input_variables = p_entry->input_variables;
    p_module->output_variable_count = p_entry->output_variable_count;
    p_module->output_variables = p_entry->output_variables;
    p_module->interface_variable_count = p_entry->interface_variable_count;
    p_module->interface_variables = p_entry->interface_variables;
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = DisambiguateStorageBufferSrvUav(p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = SynchronizeDescriptorSets(p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }
  if (result == SPV_REFLECT_RESULT_SUCCESS) {
    result = ParseExecutionModes(&parser, p_module);
    SPV_REFLECT_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  }

  // Destroy module if parse was not successful
  if (result != SPV_REFLECT_RESULT_SUCCESS) {
    spvReflectDestroyShaderModule(p_module);
  }

  DestroyParser(&parser);

  return result;
}

SpvReflectResult spvReflectCreateShaderModule(size_t size, const void* p_code, SpvReflectShaderModule* p_module) {
  return CreateShaderModule(0, size, p_code, p_module);
}

SpvReflectResult spvReflectCreateShaderModule2(uint32_t flags, size_t size, const void* p_code, SpvReflectShaderModule* p_module) {
  return CreateShaderModule(flags, size, p_code, p_module);
}

SpvReflectResult spvReflectGetShaderModule(size_t size, const void* p_code, SpvReflectShaderModule* p_module) {
  return spvReflectCreateShaderModule(size, p_code, p_module);
}

static void SafeFreeTypes(SpvReflectTypeDescription* p_type) {
  if (IsNull(p_type) || p_type->copied) {
    return;
  }

  if (IsNotNull(p_type->members)) {
    for (size_t i = 0; i < p_type->member_count; ++i) {
      SpvReflectTypeDescription* p_member = &p_type->members[i];
      SafeFreeTypes(p_member);
    }

    SafeFree(p_type->members);
    p_type->members = NULL;
  }
}

static void SafeFreeBlockVariables(SpvReflectBlockVariable* p_block) {
  if (IsNull(p_block)) {
    return;
  }

  // We share pointers to Physical Pointer structs and don't want to double free
  if (p_block->flags & SPV_REFLECT_VARIABLE_FLAGS_PHYSICAL_POINTER_COPY) {
    return;
  }

  if (IsNotNull(p_block->members)) {
    for (size_t i = 0; i < p_block->member_count; ++i) {
      SpvReflectBlockVariable* p_member = &p_block->members[i];
      SafeFreeBlockVariables(p_member);
    }

    SafeFree(p_block->members);
    p_block->members = NULL;
  }
}

static void SafeFreeInterfaceVariable(SpvReflectInterfaceVariable* p_interface) {
  if (IsNull(p_interface)) {
    return;
  }

  if (IsNotNull(p_interface->members)) {
    for (size_t i = 0; i < p_interface->member_count; ++i) {
      SpvReflectInterfaceVariable* p_member = &p_interface->members[i];
      SafeFreeInterfaceVariable(p_member);
    }

    SafeFree(p_interface->members);
    p_interface->members = NULL;
  }
}

void spvReflectDestroyShaderModule(SpvReflectShaderModule* p_module) {
  if (IsNull(p_module->_internal)) {
    return;
  }

  SafeFree(p_module->source_source);

  // Descriptor set bindings
  for (size_t i = 0; i < p_module->descriptor_set_count; ++i) {
    SpvReflectDescriptorSet* p_set = &p_module->descriptor_sets[i];
    free(p_set->bindings);
  }

  // Descriptor binding blocks
  for (size_t i = 0; i < p_module->descriptor_binding_count; ++i) {
    SpvReflectDescriptorBinding* p_descriptor = &p_module->descriptor_bindings[i];
    if (IsNotNull(p_descriptor->byte_address_buffer_offsets)) {
      SafeFree(p_descriptor->byte_address_buffer_offsets);
    }
    SafeFreeBlockVariables(&p_descriptor->block);
  }
  SafeFree(p_module->descriptor_bindings);

  // Entry points
  for (size_t i = 0; i < p_module->entry_point_count; ++i) {
    SpvReflectEntryPoint* p_entry = &p_module->entry_points[i];
    for (size_t j = 0; j < p_entry->interface_variable_count; j++) {
      SafeFreeInterfaceVariable(&p_entry->interface_variables[j]);
    }
    for (uint32_t j = 0; j < p_entry->descriptor_set_count; ++j) {
      SafeFree(p_entry->descriptor_sets[j].bindings);
    }
    SafeFree(p_entry->descriptor_sets);
    SafeFree(p_entry->input_variables);
    SafeFree(p_entry->output_variables);
    SafeFree(p_entry->interface_variables);
    SafeFree(p_entry->used_uniforms);
    SafeFree(p_entry->used_push_constants);
    SafeFree(p_entry->execution_modes);
  }
  SafeFree(p_module->capabilities);
  SafeFree(p_module->entry_points);
  SafeFree(p_module->spec_constants);

  // Push constants
  for (size_t i = 0; i < p_module->push_constant_block_count; ++i) {
    SafeFreeBlockVariables(&p_module->push_constant_blocks[i]);
  }
  SafeFree(p_module->push_constant_blocks);

  // Type infos
  for (size_t i = 0; i < p_module->_internal->type_description_count; ++i) {
    SpvReflectTypeDescription* p_type = &p_module->_internal->type_descriptions[i];
    if (IsNotNull(p_type->members)) {
      SafeFreeTypes(p_type);
    }
    SafeFree(p_type->members);
  }
  SafeFree(p_module->_internal->type_descriptions);

  // Free SPIR-V code if there was a copy
  if ((p_module->_internal->module_flags & SPV_REFLECT_MODULE_FLAG_NO_COPY) == 0) {
    SafeFree(p_module->_internal->spirv_code);
  }
  // Free internal
  SafeFree(p_module->_internal);
}

uint32_t spvReflectGetCodeSize(const SpvReflectShaderModule* p_module) {
  if (IsNull(p_module)) {
    return 0;
  }

  return (uint32_t)(p_module->_internal->spirv_size);
}

const uint32_t* spvReflectGetCode(const SpvReflectShaderModule* p_module) {
  if (IsNull(p_module)) {
    return NULL;
  }

  return p_module->_internal->spirv_code;
}

const SpvReflectEntryPoint* spvReflectGetEntryPoint(const SpvReflectShaderModule* p_module, const char* entry_point) {
  if (IsNull(p_module) || IsNull(entry_point)) {
    return NULL;
  }

  for (uint32_t i = 0; i < p_module->entry_point_count; ++i) {
    if (strcmp(p_module->entry_points[i].name, entry_point) == 0) {
      return &p_module->entry_points[i];
    }
  }
  return NULL;
}

SpvReflectResult spvReflectEnumerateDescriptorBindings(const SpvReflectShaderModule* p_module, uint32_t* p_count,
                                                       SpvReflectDescriptorBinding** pp_bindings) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  if (IsNotNull(pp_bindings)) {
    if (*p_count != p_module->descriptor_binding_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectDescriptorBinding* p_bindings = (SpvReflectDescriptorBinding*)&p_module->descriptor_bindings[index];
      pp_bindings[index] = p_bindings;
    }
  } else {
    *p_count = p_module->descriptor_binding_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateEntryPointDescriptorBindings(const SpvReflectShaderModule* p_module, const char* entry_point,
                                                                 uint32_t* p_count, SpvReflectDescriptorBinding** pp_bindings) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
  if (IsNull(p_entry)) {
    return SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
  }

  uint32_t count = 0;
  for (uint32_t i = 0; i < p_module->descriptor_binding_count; ++i) {
    bool found = SearchSortedUint32(p_entry->used_uniforms, p_entry->used_uniform_count, p_module->descriptor_bindings[i].spirv_id);
    if (found) {
      if (IsNotNull(pp_bindings)) {
        if (count >= *p_count) {
          return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
        }
        pp_bindings[count++] = (SpvReflectDescriptorBinding*)&p_module->descriptor_bindings[i];
      } else {
        ++count;
      }
    }
  }
  if (IsNotNull(pp_bindings)) {
    if (count != *p_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }
  } else {
    *p_count = count;
  }
  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateDescriptorSets(const SpvReflectShaderModule* p_module, uint32_t* p_count,
                                                   SpvReflectDescriptorSet** pp_sets) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  if (IsNotNull(pp_sets)) {
    if (*p_count != p_module->descriptor_set_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectDescriptorSet* p_set = (SpvReflectDescriptorSet*)&p_module->descriptor_sets[index];
      pp_sets[index] = p_set;
    }
  } else {
    *p_count = p_module->descriptor_set_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateEntryPointDescriptorSets(const SpvReflectShaderModule* p_module, const char* entry_point,
                                                             uint32_t* p_count, SpvReflectDescriptorSet** pp_sets) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
  if (IsNull(p_entry)) {
    return SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
  }

  if (IsNotNull(pp_sets)) {
    if (*p_count != p_entry->descriptor_set_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectDescriptorSet* p_set = (SpvReflectDescriptorSet*)&p_entry->descriptor_sets[index];
      pp_sets[index] = p_set;
    }
  } else {
    *p_count = p_entry->descriptor_set_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateInterfaceVariables(const SpvReflectShaderModule* p_module, uint32_t* p_count,
                                                       SpvReflectInterfaceVariable** pp_variables) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  if (IsNotNull(pp_variables)) {
    if (*p_count != p_module->interface_variable_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectInterfaceVariable* p_var = &p_module->interface_variables[index];
      pp_variables[index] = p_var;
    }
  } else {
    *p_count = p_module->interface_variable_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateEntryPointInterfaceVariables(const SpvReflectShaderModule* p_module, const char* entry_point,
                                                                 uint32_t* p_count, SpvReflectInterfaceVariable** pp_variables) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
  if (IsNull(p_entry)) {
    return SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
  }

  if (IsNotNull(pp_variables)) {
    if (*p_count != p_entry->interface_variable_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectInterfaceVariable* p_var = &p_entry->interface_variables[index];
      pp_variables[index] = p_var;
    }
  } else {
    *p_count = p_entry->interface_variable_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateInputVariables(const SpvReflectShaderModule* p_module, uint32_t* p_count,
                                                   SpvReflectInterfaceVariable** pp_variables) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  if (IsNotNull(pp_variables)) {
    if (*p_count != p_module->input_variable_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectInterfaceVariable* p_var = p_module->input_variables[index];
      pp_variables[index] = p_var;
    }
  } else {
    *p_count = p_module->input_variable_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateEntryPointInputVariables(const SpvReflectShaderModule* p_module, const char* entry_point,
                                                             uint32_t* p_count, SpvReflectInterfaceVariable** pp_variables) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
  if (IsNull(p_entry)) {
    return SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
  }

  if (IsNotNull(pp_variables)) {
    if (*p_count != p_entry->input_variable_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectInterfaceVariable* p_var = p_entry->input_variables[index];
      pp_variables[index] = p_var;
    }
  } else {
    *p_count = p_entry->input_variable_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateOutputVariables(const SpvReflectShaderModule* p_module, uint32_t* p_count,
                                                    SpvReflectInterfaceVariable** pp_variables) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  if (IsNotNull(pp_variables)) {
    if (*p_count != p_module->output_variable_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectInterfaceVariable* p_var = p_module->output_variables[index];
      pp_variables[index] = p_var;
    }
  } else {
    *p_count = p_module->output_variable_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateEntryPointOutputVariables(const SpvReflectShaderModule* p_module, const char* entry_point,
                                                              uint32_t* p_count, SpvReflectInterfaceVariable** pp_variables) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
  if (IsNull(p_entry)) {
    return SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
  }

  if (IsNotNull(pp_variables)) {
    if (*p_count != p_entry->output_variable_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectInterfaceVariable* p_var = p_entry->output_variables[index];
      pp_variables[index] = p_var;
    }
  } else {
    *p_count = p_entry->output_variable_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumeratePushConstantBlocks(const SpvReflectShaderModule* p_module, uint32_t* p_count,
                                                       SpvReflectBlockVariable** pp_blocks) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  if (pp_blocks != NULL) {
    if (*p_count != p_module->push_constant_block_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectBlockVariable* p_push_constant_blocks = (SpvReflectBlockVariable*)&p_module->push_constant_blocks[index];
      pp_blocks[index] = p_push_constant_blocks;
    }
  } else {
    *p_count = p_module->push_constant_block_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}
SpvReflectResult spvReflectEnumeratePushConstants(const SpvReflectShaderModule* p_module, uint32_t* p_count,
                                                  SpvReflectBlockVariable** pp_blocks) {
  return spvReflectEnumeratePushConstantBlocks(p_module, p_count, pp_blocks);
}

SpvReflectResult spvReflectEnumerateEntryPointPushConstantBlocks(const SpvReflectShaderModule* p_module, const char* entry_point,
                                                                 uint32_t* p_count, SpvReflectBlockVariable** pp_blocks) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
  if (IsNull(p_entry)) {
    return SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
  }

  uint32_t count = 0;
  for (uint32_t i = 0; i < p_module->push_constant_block_count; ++i) {
    bool found = SearchSortedUint32(p_entry->used_push_constants, p_entry->used_push_constant_count,
                                    p_module->push_constant_blocks[i].spirv_id);
    if (found) {
      if (IsNotNull(pp_blocks)) {
        if (count >= *p_count) {
          return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
        }
        pp_blocks[count++] = (SpvReflectBlockVariable*)&p_module->push_constant_blocks[i];
      } else {
        ++count;
      }
    }
  }
  if (IsNotNull(pp_blocks)) {
    if (count != *p_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }
  } else {
    *p_count = count;
  }
  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectEnumerateSpecializationConstants(const SpvReflectShaderModule* p_module, uint32_t* p_count,
                                                            SpvReflectSpecializationConstant** pp_constants) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_count)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  if (IsNotNull(pp_constants)) {
    if (*p_count != p_module->spec_constant_count) {
      return SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH;
    }

    for (uint32_t index = 0; index < *p_count; ++index) {
      SpvReflectSpecializationConstant* p_constant = (SpvReflectSpecializationConstant*)&p_module->spec_constants[index];
      pp_constants[index] = p_constant;
    }
  } else {
    *p_count = p_module->spec_constant_count;
  }

  return SPV_REFLECT_RESULT_SUCCESS;
}

const SpvReflectDescriptorBinding* spvReflectGetDescriptorBinding(const SpvReflectShaderModule* p_module, uint32_t binding_number,
                                                                  uint32_t set_number, SpvReflectResult* p_result) {
  const SpvReflectDescriptorBinding* p_descriptor = NULL;
  if (IsNotNull(p_module)) {
    for (uint32_t index = 0; index < p_module->descriptor_binding_count; ++index) {
      const SpvReflectDescriptorBinding* p_potential = &p_module->descriptor_bindings[index];
      if ((p_potential->binding == binding_number) && (p_potential->set == set_number)) {
        p_descriptor = p_potential;
        break;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_descriptor)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_descriptor;
}

const SpvReflectDescriptorBinding* spvReflectGetEntryPointDescriptorBinding(const SpvReflectShaderModule* p_module,
                                                                            const char* entry_point, uint32_t binding_number,
                                                                            uint32_t set_number, SpvReflectResult* p_result) {
  const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
  if (IsNull(p_entry)) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }
  const SpvReflectDescriptorBinding* p_descriptor = NULL;
  if (IsNotNull(p_module)) {
    for (uint32_t index = 0; index < p_module->descriptor_binding_count; ++index) {
      const SpvReflectDescriptorBinding* p_potential = &p_module->descriptor_bindings[index];
      bool found = SearchSortedUint32(p_entry->used_uniforms, p_entry->used_uniform_count, p_potential->spirv_id);
      if ((p_potential->binding == binding_number) && (p_potential->set == set_number) && found) {
        p_descriptor = p_potential;
        break;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_descriptor)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_descriptor;
}

const SpvReflectDescriptorSet* spvReflectGetDescriptorSet(const SpvReflectShaderModule* p_module, uint32_t set_number,
                                                          SpvReflectResult* p_result) {
  const SpvReflectDescriptorSet* p_set = NULL;
  if (IsNotNull(p_module)) {
    for (uint32_t index = 0; index < p_module->descriptor_set_count; ++index) {
      const SpvReflectDescriptorSet* p_potential = &p_module->descriptor_sets[index];
      if (p_potential->set == set_number) {
        p_set = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_set)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_set;
}

const SpvReflectDescriptorSet* spvReflectGetEntryPointDescriptorSet(const SpvReflectShaderModule* p_module, const char* entry_point,
                                                                    uint32_t set_number, SpvReflectResult* p_result) {
  const SpvReflectDescriptorSet* p_set = NULL;
  if (IsNotNull(p_module)) {
    const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
    if (IsNull(p_entry)) {
      if (IsNotNull(p_result)) {
        *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
      }
      return NULL;
    }
    for (uint32_t index = 0; index < p_entry->descriptor_set_count; ++index) {
      const SpvReflectDescriptorSet* p_potential = &p_entry->descriptor_sets[index];
      if (p_potential->set == set_number) {
        p_set = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_set)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_set;
}

const SpvReflectInterfaceVariable* spvReflectGetInputVariableByLocation(const SpvReflectShaderModule* p_module, uint32_t location,
                                                                        SpvReflectResult* p_result) {
  if (location == INVALID_VALUE) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }
  const SpvReflectInterfaceVariable* p_var = NULL;
  if (IsNotNull(p_module)) {
    for (uint32_t index = 0; index < p_module->input_variable_count; ++index) {
      const SpvReflectInterfaceVariable* p_potential = p_module->input_variables[index];
      if (p_potential->location == location) {
        p_var = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_var)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_var;
}
const SpvReflectInterfaceVariable* spvReflectGetInputVariable(const SpvReflectShaderModule* p_module, uint32_t location,
                                                              SpvReflectResult* p_result) {
  return spvReflectGetInputVariableByLocation(p_module, location, p_result);
}

const SpvReflectInterfaceVariable* spvReflectGetEntryPointInputVariableByLocation(const SpvReflectShaderModule* p_module,
                                                                                  const char* entry_point, uint32_t location,
                                                                                  SpvReflectResult* p_result) {
  if (location == INVALID_VALUE) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }

  const SpvReflectInterfaceVariable* p_var = NULL;
  if (IsNotNull(p_module)) {
    const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
    if (IsNull(p_entry)) {
      if (IsNotNull(p_result)) {
        *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
      }
      return NULL;
    }
    for (uint32_t index = 0; index < p_entry->input_variable_count; ++index) {
      const SpvReflectInterfaceVariable* p_potential = p_entry->input_variables[index];
      if (p_potential->location == location) {
        p_var = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_var)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_var;
}

const SpvReflectInterfaceVariable* spvReflectGetInputVariableBySemantic(const SpvReflectShaderModule* p_module,
                                                                        const char* semantic, SpvReflectResult* p_result) {
  if (IsNull(semantic)) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
    }
    return NULL;
  }
  if (semantic[0] == '\0') {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }
  const SpvReflectInterfaceVariable* p_var = NULL;
  if (IsNotNull(p_module)) {
    for (uint32_t index = 0; index < p_module->input_variable_count; ++index) {
      const SpvReflectInterfaceVariable* p_potential = p_module->input_variables[index];
      if (p_potential->semantic != NULL && strcmp(p_potential->semantic, semantic) == 0) {
        p_var = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_var)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_var;
}

const SpvReflectInterfaceVariable* spvReflectGetEntryPointInputVariableBySemantic(const SpvReflectShaderModule* p_module,
                                                                                  const char* entry_point, const char* semantic,
                                                                                  SpvReflectResult* p_result) {
  if (IsNull(semantic)) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
    }
    return NULL;
  }
  if (semantic[0] == '\0') {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }
  const SpvReflectInterfaceVariable* p_var = NULL;
  if (IsNotNull(p_module)) {
    const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
    if (IsNull(p_entry)) {
      if (IsNotNull(p_result)) {
        *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
      }
      return NULL;
    }
    for (uint32_t index = 0; index < p_entry->input_variable_count; ++index) {
      const SpvReflectInterfaceVariable* p_potential = p_entry->input_variables[index];
      if (p_potential->semantic != NULL && strcmp(p_potential->semantic, semantic) == 0) {
        p_var = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_var)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_var;
}

const SpvReflectInterfaceVariable* spvReflectGetOutputVariableByLocation(const SpvReflectShaderModule* p_module, uint32_t location,
                                                                         SpvReflectResult* p_result) {
  if (location == INVALID_VALUE) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }
  const SpvReflectInterfaceVariable* p_var = NULL;
  if (IsNotNull(p_module)) {
    for (uint32_t index = 0; index < p_module->output_variable_count; ++index) {
      const SpvReflectInterfaceVariable* p_potential = p_module->output_variables[index];
      if (p_potential->location == location) {
        p_var = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_var)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_var;
}
const SpvReflectInterfaceVariable* spvReflectGetOutputVariable(const SpvReflectShaderModule* p_module, uint32_t location,
                                                               SpvReflectResult* p_result) {
  return spvReflectGetOutputVariableByLocation(p_module, location, p_result);
}

const SpvReflectInterfaceVariable* spvReflectGetEntryPointOutputVariableByLocation(const SpvReflectShaderModule* p_module,
                                                                                   const char* entry_point, uint32_t location,
                                                                                   SpvReflectResult* p_result) {
  if (location == INVALID_VALUE) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }

  const SpvReflectInterfaceVariable* p_var = NULL;
  if (IsNotNull(p_module)) {
    const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
    if (IsNull(p_entry)) {
      if (IsNotNull(p_result)) {
        *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
      }
      return NULL;
    }
    for (uint32_t index = 0; index < p_entry->output_variable_count; ++index) {
      const SpvReflectInterfaceVariable* p_potential = p_entry->output_variables[index];
      if (p_potential->location == location) {
        p_var = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_var)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_var;
}

const SpvReflectInterfaceVariable* spvReflectGetOutputVariableBySemantic(const SpvReflectShaderModule* p_module,
                                                                         const char* semantic, SpvReflectResult* p_result) {
  if (IsNull(semantic)) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
    }
    return NULL;
  }
  if (semantic[0] == '\0') {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }
  const SpvReflectInterfaceVariable* p_var = NULL;
  if (IsNotNull(p_module)) {
    for (uint32_t index = 0; index < p_module->output_variable_count; ++index) {
      const SpvReflectInterfaceVariable* p_potential = p_module->output_variables[index];
      if (p_potential->semantic != NULL && strcmp(p_potential->semantic, semantic) == 0) {
        p_var = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_var)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_var;
}

const SpvReflectInterfaceVariable* spvReflectGetEntryPointOutputVariableBySemantic(const SpvReflectShaderModule* p_module,
                                                                                   const char* entry_point, const char* semantic,
                                                                                   SpvReflectResult* p_result) {
  if (IsNull(semantic)) {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
    }
    return NULL;
  }
  if (semantic[0] == '\0') {
    if (IsNotNull(p_result)) {
      *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
    }
    return NULL;
  }
  const SpvReflectInterfaceVariable* p_var = NULL;
  if (IsNotNull(p_module)) {
    const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
    if (IsNull(p_entry)) {
      if (IsNotNull(p_result)) {
        *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
      }
      return NULL;
    }
    for (uint32_t index = 0; index < p_entry->output_variable_count; ++index) {
      const SpvReflectInterfaceVariable* p_potential = p_entry->output_variables[index];
      if (p_potential->semantic != NULL && strcmp(p_potential->semantic, semantic) == 0) {
        p_var = p_potential;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_var)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_var;
}

const SpvReflectBlockVariable* spvReflectGetPushConstantBlock(const SpvReflectShaderModule* p_module, uint32_t index,
                                                              SpvReflectResult* p_result) {
  const SpvReflectBlockVariable* p_push_constant = NULL;
  if (IsNotNull(p_module)) {
    if (index < p_module->push_constant_block_count) {
      p_push_constant = &p_module->push_constant_blocks[index];
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_push_constant)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_push_constant;
}
const SpvReflectBlockVariable* spvReflectGetPushConstant(const SpvReflectShaderModule* p_module, uint32_t index,
                                                         SpvReflectResult* p_result) {
  return spvReflectGetPushConstantBlock(p_module, index, p_result);
}

const SpvReflectBlockVariable* spvReflectGetEntryPointPushConstantBlock(const SpvReflectShaderModule* p_module,
                                                                        const char* entry_point, SpvReflectResult* p_result) {
  const SpvReflectBlockVariable* p_push_constant = NULL;
  if (IsNotNull(p_module)) {
    const SpvReflectEntryPoint* p_entry = spvReflectGetEntryPoint(p_module, entry_point);
    if (IsNull(p_entry)) {
      if (IsNotNull(p_result)) {
        *p_result = SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
      }
      return NULL;
    }
    for (uint32_t i = 0; i < p_module->push_constant_block_count; ++i) {
      bool found = SearchSortedUint32(p_entry->used_push_constants, p_entry->used_push_constant_count,
                                      p_module->push_constant_blocks[i].spirv_id);
      if (found) {
        p_push_constant = &p_module->push_constant_blocks[i];
        break;
      }
    }
  }
  if (IsNotNull(p_result)) {
    *p_result = IsNotNull(p_push_constant)
                    ? SPV_REFLECT_RESULT_SUCCESS
                    : (IsNull(p_module) ? SPV_REFLECT_RESULT_ERROR_NULL_POINTER : SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND);
  }
  return p_push_constant;
}

SpvReflectResult spvReflectChangeDescriptorBindingNumbers(SpvReflectShaderModule* p_module,
                                                          const SpvReflectDescriptorBinding* p_binding, uint32_t new_binding_number,
                                                          uint32_t new_set_binding) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_binding)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }

  SpvReflectDescriptorBinding* p_target_descriptor = NULL;
  for (uint32_t index = 0; index < p_module->descriptor_binding_count; ++index) {
    if (&p_module->descriptor_bindings[index] == p_binding) {
      p_target_descriptor = &p_module->descriptor_bindings[index];
      break;
    }
  }

  if (IsNotNull(p_target_descriptor)) {
    if (p_target_descriptor->word_offset.binding > (p_module->_internal->spirv_word_count - 1)) {
      return SPV_REFLECT_RESULT_ERROR_RANGE_EXCEEDED;
    }
    // Binding number
    if (new_binding_number != (uint32_t)SPV_REFLECT_BINDING_NUMBER_DONT_CHANGE) {
      uint32_t* p_code = p_module->_internal->spirv_code + p_target_descriptor->word_offset.binding;
      *p_code = new_binding_number;
      p_target_descriptor->binding = new_binding_number;
    }
    // Set number
    if (new_set_binding != (uint32_t)SPV_REFLECT_SET_NUMBER_DONT_CHANGE) {
      uint32_t* p_code = p_module->_internal->spirv_code + p_target_descriptor->word_offset.set;
      *p_code = new_set_binding;
      p_target_descriptor->set = new_set_binding;
    }
  }

  SpvReflectResult result = SPV_REFLECT_RESULT_SUCCESS;
  if (new_set_binding != (uint32_t)SPV_REFLECT_SET_NUMBER_DONT_CHANGE) {
    result = SynchronizeDescriptorSets(p_module);
  }
  return result;
}
SpvReflectResult spvReflectChangeDescriptorBindingNumber(SpvReflectShaderModule* p_module,
                                                         const SpvReflectDescriptorBinding* p_descriptor_binding,
                                                         uint32_t new_binding_number, uint32_t optional_new_set_number) {
  return spvReflectChangeDescriptorBindingNumbers(p_module, p_descriptor_binding, new_binding_number, optional_new_set_number);
}

SpvReflectResult spvReflectChangeDescriptorSetNumber(SpvReflectShaderModule* p_module, const SpvReflectDescriptorSet* p_set,
                                                     uint32_t new_set_number) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_set)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  SpvReflectDescriptorSet* p_target_set = NULL;
  for (uint32_t index = 0; index < SPV_REFLECT_MAX_DESCRIPTOR_SETS; ++index) {
    // The descriptor sets for specific entry points might not be in this set,
    // so just match on set index.
    if (p_module->descriptor_sets[index].set == p_set->set) {
      p_target_set = (SpvReflectDescriptorSet*)p_set;
      break;
    }
  }

  SpvReflectResult result = SPV_REFLECT_RESULT_SUCCESS;
  if (IsNotNull(p_target_set) && new_set_number != (uint32_t)SPV_REFLECT_SET_NUMBER_DONT_CHANGE) {
    for (uint32_t index = 0; index < p_target_set->binding_count; ++index) {
      SpvReflectDescriptorBinding* p_descriptor = p_target_set->bindings[index];
      if (p_descriptor->word_offset.set > (p_module->_internal->spirv_word_count - 1)) {
        return SPV_REFLECT_RESULT_ERROR_RANGE_EXCEEDED;
      }

      uint32_t* p_code = p_module->_internal->spirv_code + p_descriptor->word_offset.set;
      *p_code = new_set_number;
      p_descriptor->set = new_set_number;
    }

    result = SynchronizeDescriptorSets(p_module);
  }

  return result;
}

static SpvReflectResult ChangeVariableLocation(SpvReflectShaderModule* p_module, SpvReflectInterfaceVariable* p_variable,
                                               uint32_t new_location) {
  if (p_variable->word_offset.location > (p_module->_internal->spirv_word_count - 1)) {
    return SPV_REFLECT_RESULT_ERROR_RANGE_EXCEEDED;
  }
  uint32_t* p_code = p_module->_internal->spirv_code + p_variable->word_offset.location;
  *p_code = new_location;
  p_variable->location = new_location;
  return SPV_REFLECT_RESULT_SUCCESS;
}

SpvReflectResult spvReflectChangeInputVariableLocation(SpvReflectShaderModule* p_module,
                                                       const SpvReflectInterfaceVariable* p_input_variable, uint32_t new_location) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_input_variable)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  for (uint32_t index = 0; index < p_module->input_variable_count; ++index) {
    if (p_module->input_variables[index] == p_input_variable) {
      return ChangeVariableLocation(p_module, p_module->input_variables[index], new_location);
    }
  }
  return SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
}

SpvReflectResult spvReflectChangeOutputVariableLocation(SpvReflectShaderModule* p_module,
                                                        const SpvReflectInterfaceVariable* p_output_variable,
                                                        uint32_t new_location) {
  if (IsNull(p_module)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  if (IsNull(p_output_variable)) {
    return SPV_REFLECT_RESULT_ERROR_NULL_POINTER;
  }
  for (uint32_t index = 0; index < p_module->output_variable_count; ++index) {
    if (p_module->output_variables[index] == p_output_variable) {
      return ChangeVariableLocation(p_module, p_module->output_variables[index], new_location);
    }
  }
  return SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND;
}

const char* spvReflectSourceLanguage(SpvSourceLanguage source_lang) {
  switch (source_lang) {
    case SpvSourceLanguageESSL:
      return "ESSL";
    case SpvSourceLanguageGLSL:
      return "GLSL";
    case SpvSourceLanguageOpenCL_C:
      return "OpenCL_C";
    case SpvSourceLanguageOpenCL_CPP:
      return "OpenCL_CPP";
    case SpvSourceLanguageHLSL:
      return "HLSL";
    case SpvSourceLanguageCPP_for_OpenCL:
      return "CPP_for_OpenCL";
    case SpvSourceLanguageSYCL:
      return "SYCL";
    case SpvSourceLanguageHERO_C:
      return "Hero C";
    case SpvSourceLanguageNZSL:
      return "NZSL";
    default:
      break;
  }
  // The source language is SpvSourceLanguageUnknown, SpvSourceLanguageMax, or
  // some other value that does not correspond to a knonwn language.
  return "Unknown";
}

const char* spvReflectBlockVariableTypeName(const SpvReflectBlockVariable* p_var) {
  if (p_var == NULL) {
    return NULL;
  }
  return p_var->type_description->type_name;
}
