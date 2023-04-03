/*
 * Mesa 3-D graphics library
 *
 * Copyright © 2015 Red Hat
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "shader_enums.h"
#include "util/macros.h"
#include "mesa/main/config.h"

#define ENUM(x) [x] = #x
#define NAME(val) ((((val) < ARRAY_SIZE(names)) && names[(val)]) ? names[(val)] : "UNKNOWN")

const char *
gl_shader_stage_name(gl_shader_stage stage)
{
   static const char *names[] = {
      ENUM(MESA_SHADER_VERTEX),
      ENUM(MESA_SHADER_TESS_CTRL),
      ENUM(MESA_SHADER_TESS_EVAL),
      ENUM(MESA_SHADER_GEOMETRY),
      ENUM(MESA_SHADER_FRAGMENT),
      ENUM(MESA_SHADER_COMPUTE),
      ENUM(MESA_SHADER_TASK),
      ENUM(MESA_SHADER_MESH),
      ENUM(MESA_SHADER_RAYGEN),
      ENUM(MESA_SHADER_ANY_HIT),
      ENUM(MESA_SHADER_CLOSEST_HIT),
      ENUM(MESA_SHADER_MISS),
      ENUM(MESA_SHADER_INTERSECTION),
      ENUM(MESA_SHADER_CALLABLE),
      ENUM(MESA_SHADER_KERNEL),
   };
   STATIC_ASSERT(ARRAY_SIZE(names) == MESA_ALL_SHADER_STAGES);
   return NAME(stage);
}

/**
 * Translate a gl_shader_stage to a short shader stage name for debug
 * printouts and error messages.
 */
const char *
_mesa_shader_stage_to_string(unsigned stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:   return "vertex";
   case MESA_SHADER_FRAGMENT: return "fragment";
   case MESA_SHADER_GEOMETRY: return "geometry";
   case MESA_SHADER_COMPUTE:  return "compute";
   case MESA_SHADER_KERNEL:   return "kernel";
   case MESA_SHADER_TESS_CTRL: return "tessellation control";
   case MESA_SHADER_TESS_EVAL: return "tessellation evaluation";
   case MESA_SHADER_TASK:         return "task";
   case MESA_SHADER_MESH:         return "mesh";
   case MESA_SHADER_RAYGEN:       return "raygen";
   case MESA_SHADER_ANY_HIT:      return "any hit";
   case MESA_SHADER_CLOSEST_HIT:  return "closest hit";
   case MESA_SHADER_MISS:         return "miss";
   case MESA_SHADER_INTERSECTION: return "intersection";
   case MESA_SHADER_CALLABLE:     return "callable";
   }

   unreachable("Unknown shader stage.");
}

/**
 * Translate a gl_shader_stage to a shader stage abbreviation (VS, GS, FS)
 * for debug printouts and error messages.
 */
const char *
_mesa_shader_stage_to_abbrev(unsigned stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:   return "VS";
   case MESA_SHADER_FRAGMENT: return "FS";
   case MESA_SHADER_GEOMETRY: return "GS";
   case MESA_SHADER_COMPUTE:  return "CS";
   case MESA_SHADER_KERNEL:   return "CL";
   case MESA_SHADER_TESS_CTRL: return "TCS";
   case MESA_SHADER_TESS_EVAL: return "TES";
   case MESA_SHADER_TASK:         return "TASK";
   case MESA_SHADER_MESH:         return "MESH";
   case MESA_SHADER_RAYGEN:       return "RGEN";
   case MESA_SHADER_ANY_HIT:      return "RAHIT";
   case MESA_SHADER_CLOSEST_HIT:  return "RCHIT";
   case MESA_SHADER_MISS:         return "RMISS";
   case MESA_SHADER_INTERSECTION: return "RINT";
   case MESA_SHADER_CALLABLE:     return "RCALL";
   }

   unreachable("Unknown shader stage.");
}

const char *
gl_vert_attrib_name(gl_vert_attrib attrib)
{
   static const char *names[] = {
      ENUM(VERT_ATTRIB_POS),
      ENUM(VERT_ATTRIB_NORMAL),
      ENUM(VERT_ATTRIB_COLOR0),
      ENUM(VERT_ATTRIB_COLOR1),
      ENUM(VERT_ATTRIB_FOG),
      ENUM(VERT_ATTRIB_COLOR_INDEX),
      ENUM(VERT_ATTRIB_TEX0),
      ENUM(VERT_ATTRIB_TEX1),
      ENUM(VERT_ATTRIB_TEX2),
      ENUM(VERT_ATTRIB_TEX3),
      ENUM(VERT_ATTRIB_TEX4),
      ENUM(VERT_ATTRIB_TEX5),
      ENUM(VERT_ATTRIB_TEX6),
      ENUM(VERT_ATTRIB_TEX7),
      ENUM(VERT_ATTRIB_POINT_SIZE),
      ENUM(VERT_ATTRIB_GENERIC0),
      ENUM(VERT_ATTRIB_GENERIC1),
      ENUM(VERT_ATTRIB_GENERIC2),
      ENUM(VERT_ATTRIB_GENERIC3),
      ENUM(VERT_ATTRIB_GENERIC4),
      ENUM(VERT_ATTRIB_GENERIC5),
      ENUM(VERT_ATTRIB_GENERIC6),
      ENUM(VERT_ATTRIB_GENERIC7),
      ENUM(VERT_ATTRIB_GENERIC8),
      ENUM(VERT_ATTRIB_GENERIC9),
      ENUM(VERT_ATTRIB_GENERIC10),
      ENUM(VERT_ATTRIB_GENERIC11),
      ENUM(VERT_ATTRIB_GENERIC12),
      ENUM(VERT_ATTRIB_GENERIC13),
      ENUM(VERT_ATTRIB_GENERIC14),
      ENUM(VERT_ATTRIB_GENERIC15),
      ENUM(VERT_ATTRIB_EDGEFLAG),
   };
   STATIC_ASSERT(ARRAY_SIZE(names) == VERT_ATTRIB_MAX);
   return NAME(attrib);
}

const char *
gl_varying_slot_name_for_stage(gl_varying_slot slot, gl_shader_stage stage)
{
   if (stage != MESA_SHADER_FRAGMENT && slot == VARYING_SLOT_PRIMITIVE_SHADING_RATE)
      return "VARYING_SLOT_PRIMITIVE_SHADING_RATE";

   switch (stage) {
   case MESA_SHADER_MESH:
      switch (slot) {
      case VARYING_SLOT_PRIMITIVE_COUNT: return "VARYING_SLOT_PRIMITIVE_COUNT";
      case VARYING_SLOT_PRIMITIVE_INDICES: return "VARYING_SLOT_PRIMITIVE_INDICES";
      case VARYING_SLOT_CULL_PRIMITIVE: return "VARYING_SLOT_CULL_PRIMITIVE";
      default:
         /* Not an overlapping value. */
         break;
      }
      break;

   case MESA_SHADER_TASK:
      switch (slot) {
      case VARYING_SLOT_TASK_COUNT: return "VARYING_SLOT_TASK_COUNT";
      default:
         /* Not an overlapping value. */
         break;
      }
      break;

   default:
      break;
   }

   static const char *names[] = {
      ENUM(VARYING_SLOT_POS),
      ENUM(VARYING_SLOT_COL0),
      ENUM(VARYING_SLOT_COL1),
      ENUM(VARYING_SLOT_FOGC),
      ENUM(VARYING_SLOT_TEX0),
      ENUM(VARYING_SLOT_TEX1),
      ENUM(VARYING_SLOT_TEX2),
      ENUM(VARYING_SLOT_TEX3),
      ENUM(VARYING_SLOT_TEX4),
      ENUM(VARYING_SLOT_TEX5),
      ENUM(VARYING_SLOT_TEX6),
      ENUM(VARYING_SLOT_TEX7),
      ENUM(VARYING_SLOT_PSIZ),
      ENUM(VARYING_SLOT_BFC0),
      ENUM(VARYING_SLOT_BFC1),
      ENUM(VARYING_SLOT_EDGE),
      ENUM(VARYING_SLOT_CLIP_VERTEX),
      ENUM(VARYING_SLOT_CLIP_DIST0),
      ENUM(VARYING_SLOT_CLIP_DIST1),
      ENUM(VARYING_SLOT_CULL_DIST0),
      ENUM(VARYING_SLOT_CULL_DIST1),
      ENUM(VARYING_SLOT_PRIMITIVE_ID),
      ENUM(VARYING_SLOT_LAYER),
      ENUM(VARYING_SLOT_VIEWPORT),
      ENUM(VARYING_SLOT_FACE),
      ENUM(VARYING_SLOT_PNTC),
      ENUM(VARYING_SLOT_TESS_LEVEL_OUTER),
      ENUM(VARYING_SLOT_TESS_LEVEL_INNER),
      ENUM(VARYING_SLOT_BOUNDING_BOX0),
      ENUM(VARYING_SLOT_BOUNDING_BOX1),
      ENUM(VARYING_SLOT_VIEW_INDEX),
      ENUM(VARYING_SLOT_VIEWPORT_MASK),
      ENUM(VARYING_SLOT_VAR0),
      ENUM(VARYING_SLOT_VAR1),
      ENUM(VARYING_SLOT_VAR2),
      ENUM(VARYING_SLOT_VAR3),
      ENUM(VARYING_SLOT_VAR4),
      ENUM(VARYING_SLOT_VAR5),
      ENUM(VARYING_SLOT_VAR6),
      ENUM(VARYING_SLOT_VAR7),
      ENUM(VARYING_SLOT_VAR8),
      ENUM(VARYING_SLOT_VAR9),
      ENUM(VARYING_SLOT_VAR10),
      ENUM(VARYING_SLOT_VAR11),
      ENUM(VARYING_SLOT_VAR12),
      ENUM(VARYING_SLOT_VAR13),
      ENUM(VARYING_SLOT_VAR14),
      ENUM(VARYING_SLOT_VAR15),
      ENUM(VARYING_SLOT_VAR16),
      ENUM(VARYING_SLOT_VAR17),
      ENUM(VARYING_SLOT_VAR18),
      ENUM(VARYING_SLOT_VAR19),
      ENUM(VARYING_SLOT_VAR20),
      ENUM(VARYING_SLOT_VAR21),
      ENUM(VARYING_SLOT_VAR22),
      ENUM(VARYING_SLOT_VAR23),
      ENUM(VARYING_SLOT_VAR24),
      ENUM(VARYING_SLOT_VAR25),
      ENUM(VARYING_SLOT_VAR26),
      ENUM(VARYING_SLOT_VAR27),
      ENUM(VARYING_SLOT_VAR28),
      ENUM(VARYING_SLOT_VAR29),
      ENUM(VARYING_SLOT_VAR30),
      ENUM(VARYING_SLOT_VAR31),
   };
   STATIC_ASSERT(ARRAY_SIZE(names) == VARYING_SLOT_MAX);
   return NAME(slot);
}

const char *
gl_system_value_name(gl_system_value sysval)
{
   static const char *names[] = {
     ENUM(SYSTEM_VALUE_SUBGROUP_SIZE),
     ENUM(SYSTEM_VALUE_SUBGROUP_INVOCATION),
     ENUM(SYSTEM_VALUE_SUBGROUP_EQ_MASK),
     ENUM(SYSTEM_VALUE_SUBGROUP_GE_MASK),
     ENUM(SYSTEM_VALUE_SUBGROUP_GT_MASK),
     ENUM(SYSTEM_VALUE_SUBGROUP_LE_MASK),
     ENUM(SYSTEM_VALUE_SUBGROUP_LT_MASK),
     ENUM(SYSTEM_VALUE_NUM_SUBGROUPS),
     ENUM(SYSTEM_VALUE_SUBGROUP_ID),
     ENUM(SYSTEM_VALUE_VERTEX_ID),
     ENUM(SYSTEM_VALUE_INSTANCE_ID),
     ENUM(SYSTEM_VALUE_INSTANCE_INDEX),
     ENUM(SYSTEM_VALUE_VERTEX_ID_ZERO_BASE),
     ENUM(SYSTEM_VALUE_BASE_VERTEX),
     ENUM(SYSTEM_VALUE_FIRST_VERTEX),
     ENUM(SYSTEM_VALUE_IS_INDEXED_DRAW),
     ENUM(SYSTEM_VALUE_BASE_INSTANCE),
     ENUM(SYSTEM_VALUE_DRAW_ID),
     ENUM(SYSTEM_VALUE_INVOCATION_ID),
     ENUM(SYSTEM_VALUE_FRAG_COORD),
     ENUM(SYSTEM_VALUE_POINT_COORD),
     ENUM(SYSTEM_VALUE_LINE_COORD),
     ENUM(SYSTEM_VALUE_FRONT_FACE),
     ENUM(SYSTEM_VALUE_SAMPLE_ID),
     ENUM(SYSTEM_VALUE_SAMPLE_POS),
     ENUM(SYSTEM_VALUE_SAMPLE_MASK_IN),
     ENUM(SYSTEM_VALUE_HELPER_INVOCATION),
     ENUM(SYSTEM_VALUE_COLOR0),
     ENUM(SYSTEM_VALUE_COLOR1),
     ENUM(SYSTEM_VALUE_TESS_COORD),
     ENUM(SYSTEM_VALUE_VERTICES_IN),
     ENUM(SYSTEM_VALUE_PRIMITIVE_ID),
     ENUM(SYSTEM_VALUE_TESS_LEVEL_OUTER),
     ENUM(SYSTEM_VALUE_TESS_LEVEL_INNER),
     ENUM(SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT),
     ENUM(SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT),
     ENUM(SYSTEM_VALUE_LOCAL_INVOCATION_ID),
     ENUM(SYSTEM_VALUE_LOCAL_INVOCATION_INDEX),
     ENUM(SYSTEM_VALUE_GLOBAL_INVOCATION_ID),
     ENUM(SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID),
     ENUM(SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX),
     ENUM(SYSTEM_VALUE_WORKGROUP_ID),
     ENUM(SYSTEM_VALUE_NUM_WORKGROUPS),
     ENUM(SYSTEM_VALUE_WORKGROUP_SIZE),
     ENUM(SYSTEM_VALUE_GLOBAL_GROUP_SIZE),
     ENUM(SYSTEM_VALUE_USER_DATA_AMD),
     ENUM(SYSTEM_VALUE_WORK_DIM),
     ENUM(SYSTEM_VALUE_DEVICE_INDEX),
     ENUM(SYSTEM_VALUE_VIEW_INDEX),
     ENUM(SYSTEM_VALUE_VERTEX_CNT),
     ENUM(SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL),
     ENUM(SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE),
     ENUM(SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID),
     ENUM(SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW),
     ENUM(SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL),
     ENUM(SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID),
     ENUM(SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE),
     ENUM(SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL),
     ENUM(SYSTEM_VALUE_RAY_LAUNCH_ID),
     ENUM(SYSTEM_VALUE_RAY_LAUNCH_SIZE),
     ENUM(SYSTEM_VALUE_RAY_LAUNCH_SIZE_ADDR_AMD),
     ENUM(SYSTEM_VALUE_RAY_WORLD_ORIGIN),
     ENUM(SYSTEM_VALUE_RAY_WORLD_DIRECTION),
     ENUM(SYSTEM_VALUE_RAY_OBJECT_ORIGIN),
     ENUM(SYSTEM_VALUE_RAY_OBJECT_DIRECTION),
     ENUM(SYSTEM_VALUE_RAY_T_MIN),
     ENUM(SYSTEM_VALUE_RAY_T_MAX),
     ENUM(SYSTEM_VALUE_RAY_OBJECT_TO_WORLD),
     ENUM(SYSTEM_VALUE_RAY_WORLD_TO_OBJECT),
     ENUM(SYSTEM_VALUE_RAY_HIT_KIND),
     ENUM(SYSTEM_VALUE_RAY_FLAGS),
     ENUM(SYSTEM_VALUE_RAY_GEOMETRY_INDEX),
     ENUM(SYSTEM_VALUE_CULL_MASK),
     ENUM(SYSTEM_VALUE_MESH_VIEW_COUNT),
     ENUM(SYSTEM_VALUE_MESH_VIEW_INDICES),
     ENUM(SYSTEM_VALUE_GS_HEADER_IR3),
     ENUM(SYSTEM_VALUE_TCS_HEADER_IR3),
     ENUM(SYSTEM_VALUE_REL_PATCH_ID_IR3),
     ENUM(SYSTEM_VALUE_FRAG_SHADING_RATE),
   };
   STATIC_ASSERT(ARRAY_SIZE(names) == SYSTEM_VALUE_MAX);
   return NAME(sysval);
}

const char *
glsl_interp_mode_name(enum glsl_interp_mode qual)
{
   static const char *names[] = {
      ENUM(INTERP_MODE_NONE),
      ENUM(INTERP_MODE_SMOOTH),
      ENUM(INTERP_MODE_FLAT),
      ENUM(INTERP_MODE_NOPERSPECTIVE),
      ENUM(INTERP_MODE_EXPLICIT),
      ENUM(INTERP_MODE_COLOR),
   };
   STATIC_ASSERT(ARRAY_SIZE(names) == INTERP_MODE_COUNT);
   return NAME(qual);
}

const char *
gl_frag_result_name(gl_frag_result result)
{
   static const char *names[] = {
      ENUM(FRAG_RESULT_DEPTH),
      ENUM(FRAG_RESULT_STENCIL),
      ENUM(FRAG_RESULT_COLOR),
      ENUM(FRAG_RESULT_SAMPLE_MASK),
      ENUM(FRAG_RESULT_DATA0),
      ENUM(FRAG_RESULT_DATA1),
      ENUM(FRAG_RESULT_DATA2),
      ENUM(FRAG_RESULT_DATA3),
      ENUM(FRAG_RESULT_DATA4),
      ENUM(FRAG_RESULT_DATA5),
      ENUM(FRAG_RESULT_DATA6),
      ENUM(FRAG_RESULT_DATA7),
   };
   STATIC_ASSERT(ARRAY_SIZE(names) == FRAG_RESULT_MAX);
   return NAME(result);
}

unsigned
num_mesh_vertices_per_primitive(unsigned prim)
{
   switch (prim) {
      case SHADER_PRIM_POINTS:
         return 1;
      case SHADER_PRIM_LINES:
         return 2;
      case SHADER_PRIM_TRIANGLES:
         return 3;
      default:
         unreachable("invalid mesh shader primitive type");
   }
}
