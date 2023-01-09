#
# Copyright (C) 2018 Red Hat
# Copyright (C) 2014 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

# This file defines all the available intrinsics in one place.
#
# The Intrinsic class corresponds one-to-one with nir_intrinsic_info
# structure.

src0 = ('src', 0)
src1 = ('src', 1)
src2 = ('src', 2)
src3 = ('src', 3)
src4 = ('src', 4)

class Index(object):
    def __init__(self, c_data_type, name):
        self.c_data_type = c_data_type
        self.name = name

class Intrinsic(object):
   """Class that represents all the information about an intrinsic opcode.
   NOTE: this must be kept in sync with nir_intrinsic_info.
   """
   def __init__(self, name, src_components, dest_components,
                indices, flags, sysval, bit_sizes):
       """Parameters:

       - name: the intrinsic name
       - src_components: list of the number of components per src, 0 means
         vectorized instruction with number of components given in the
         num_components field in nir_intrinsic_instr.
       - dest_components: number of destination components, -1 means no
         dest, 0 means number of components given in num_components field
         in nir_intrinsic_instr.
       - indices: list of constant indicies
       - flags: list of semantic flags
       - sysval: is this a system-value intrinsic
       - bit_sizes: allowed dest bit_sizes or the source it must match
       """
       assert isinstance(name, str)
       assert isinstance(src_components, list)
       if src_components:
           assert isinstance(src_components[0], int)
       assert isinstance(dest_components, int)
       assert isinstance(indices, list)
       if indices:
           assert isinstance(indices[0], Index)
       assert isinstance(flags, list)
       if flags:
           assert isinstance(flags[0], str)
       assert isinstance(sysval, bool)
       if isinstance(bit_sizes, list):
           assert not bit_sizes or isinstance(bit_sizes[0], int)
       else:
           assert isinstance(bit_sizes, tuple)
           assert bit_sizes[0] == 'src'
           assert isinstance(bit_sizes[1], int)

       self.name = name
       self.num_srcs = len(src_components)
       self.src_components = src_components
       self.has_dest = (dest_components >= 0)
       self.dest_components = dest_components
       self.num_indices = len(indices)
       self.indices = indices
       self.flags = flags
       self.sysval = sysval
       self.bit_sizes = bit_sizes if isinstance(bit_sizes, list) else []
       self.bit_size_src = bit_sizes[1] if isinstance(bit_sizes, tuple) else -1

#
# Possible flags:
#

CAN_ELIMINATE = "NIR_INTRINSIC_CAN_ELIMINATE"
CAN_REORDER   = "NIR_INTRINSIC_CAN_REORDER"

INTR_INDICES = []
INTR_OPCODES = {}

def index(c_data_type, name):
    idx = Index(c_data_type, name)
    INTR_INDICES.append(idx)
    globals()[name.upper()] = idx

# Defines a new NIR intrinsic.  By default, the intrinsic will have no sources
# and no destination.
#
# You can set dest_comp=n to enable a destination for the intrinsic, in which
# case it will have that many components, or =0 for "as many components as the
# NIR destination value."
#
# Set src_comp=n to enable sources for the intruction.  It can be an array of
# component counts, or (for convenience) a scalar component count if there's
# only one source.  If a component count is 0, it will be as many components as
# the intrinsic has based on the dest_comp.
def intrinsic(name, src_comp=[], dest_comp=-1, indices=[],
              flags=[], sysval=False, bit_sizes=[]):
    assert name not in INTR_OPCODES
    INTR_OPCODES[name] = Intrinsic(name, src_comp, dest_comp,
                                   indices, flags, sysval, bit_sizes)

#
# Possible indices:
#

# Generally instructions that take a offset src argument, can encode
# a constant 'base' value which is added to the offset.
index("int", "base")

# For store instructions, a writemask for the store.
index("unsigned", "write_mask")

# The stream-id for GS emit_vertex/end_primitive intrinsics.
index("unsigned", "stream_id")

# The clip-plane id for load_user_clip_plane intrinsic.
index("unsigned", "ucp_id")

# The offset to the start of the NIR_INTRINSIC_RANGE.  This is an alternative
# to NIR_INTRINSIC_BASE for describing the valid range in intrinsics that don't
# have the implicit addition of a base to the offset.
#
# If the [range_base, range] is [0, ~0], then we don't know the possible
# range of the access.
index("unsigned", "range_base")

# The amount of data, starting from BASE or RANGE_BASE, that this
# instruction may access.  This is used to provide bounds if the offset is
# not constant.
index("unsigned", "range")

# The Vulkan descriptor set for vulkan_resource_index intrinsic.
index("unsigned", "desc_set")

# The Vulkan descriptor set binding for vulkan_resource_index intrinsic.
index("unsigned", "binding")

# Component offset
index("unsigned", "component")

# Column index for matrix system values
index("unsigned", "column")

# Interpolation mode (only meaningful for FS inputs)
index("unsigned", "interp_mode")

# A binary nir_op to use when performing a reduction or scan operation
index("unsigned", "reduction_op")

# Cluster size for reduction operations
index("unsigned", "cluster_size")

# Parameter index for a load_param intrinsic
index("unsigned", "param_idx")

# Image dimensionality for image intrinsics
index("enum glsl_sampler_dim", "image_dim")

# Non-zero if we are accessing an array image
index("bool", "image_array")

# Image format for image intrinsics
index("enum pipe_format", "format")

# Access qualifiers for image and memory access intrinsics. ACCESS_RESTRICT is
# not set at the intrinsic if the NIR was created from SPIR-V.
index("enum gl_access_qualifier", "access")

# call index for split raytracing shaders
index("unsigned", "call_idx")

# The stack size increment/decrement for split raytracing shaders
index("unsigned", "stack_size")

# Alignment for offsets and addresses
#
# These two parameters, specify an alignment in terms of a multiplier and
# an offset.  The multiplier is always a power of two.  The offset or
# address parameter X of the intrinsic is guaranteed to satisfy the
# following:
#
#                (X - align_offset) % align_mul == 0
#
# For constant offset values, align_mul will be NIR_ALIGN_MUL_MAX and the
# align_offset will be modulo that.
index("unsigned", "align_mul")
index("unsigned", "align_offset")

# The Vulkan descriptor type for a vulkan_resource_[re]index intrinsic.
index("unsigned", "desc_type")

# The nir_alu_type of input data to a store or conversion
index("nir_alu_type", "src_type")

# The nir_alu_type of the data output from a load or conversion
index("nir_alu_type", "dest_type")

# The swizzle mask for quad_swizzle_amd & masked_swizzle_amd
index("unsigned", "swizzle_mask")

# Offsets for load_shared2_amd/store_shared2_amd
index("uint8_t", "offset0")
index("uint8_t", "offset1")

# If true, both offsets have an additional stride of 64 dwords (ie. they are multiplied by 256 bytes
# in hardware, instead of 4).
index("bool", "st64")

# When set, range analysis will use it for nir_unsigned_upper_bound
index("unsigned", "arg_upper_bound_u32_amd")

# Separate source/dest access flags for copies
index("enum gl_access_qualifier", "dst_access")
index("enum gl_access_qualifier", "src_access")

# Driver location of attribute
index("unsigned", "driver_location")

# Ordering and visibility of a memory operation
index("nir_memory_semantics", "memory_semantics")

# Modes affected by a memory operation
index("nir_variable_mode", "memory_modes")

# Scope of a memory operation
index("nir_scope", "memory_scope")

# Scope of a control barrier
index("nir_scope", "execution_scope")

# Semantics of an IO instruction
index("struct nir_io_semantics", "io_semantics")

# Transform feedback info
index("struct nir_io_xfb", "io_xfb")
index("struct nir_io_xfb", "io_xfb2")

# Ray query values accessible from the RayQueryKHR object
index("nir_ray_query_value", "ray_query_value")

# Rounding mode for conversions
index("nir_rounding_mode", "rounding_mode")

# Whether or not to saturate in conversions
index("unsigned", "saturate")

# Whether or not trace_ray_intel is synchronous
index("bool", "synchronous")

# Value ID to identify SSA value loaded/stored on the stack
index("unsigned", "value_id")

# Whether to sign-extend offsets in address arithmatic (else zero extend)
index("bool", "sign_extend")

intrinsic("nop", flags=[CAN_ELIMINATE])

intrinsic("convert_alu_types", dest_comp=0, src_comp=[0],
          indices=[SRC_TYPE, DEST_TYPE, ROUNDING_MODE, SATURATE],
          flags=[CAN_ELIMINATE, CAN_REORDER])

intrinsic("load_param", dest_comp=0, indices=[PARAM_IDX], flags=[CAN_ELIMINATE])

intrinsic("load_deref", dest_comp=0, src_comp=[-1],
          indices=[ACCESS], flags=[CAN_ELIMINATE])
intrinsic("store_deref", src_comp=[-1, 0], indices=[WRITE_MASK, ACCESS])
intrinsic("copy_deref", src_comp=[-1, -1], indices=[DST_ACCESS, SRC_ACCESS])
intrinsic("memcpy_deref", src_comp=[-1, -1, 1], indices=[DST_ACCESS, SRC_ACCESS])

# Interpolation of input.  The interp_deref_at* intrinsics are similar to the
# load_var intrinsic acting on a shader input except that they interpolate the
# input differently.  The at_sample, at_offset and at_vertex intrinsics take an
# additional source that is an integer sample id, a vec2 position offset, or a
# vertex ID respectively.

intrinsic("interp_deref_at_centroid", dest_comp=0, src_comp=[1],
          flags=[ CAN_ELIMINATE, CAN_REORDER])
intrinsic("interp_deref_at_sample", src_comp=[1, 1], dest_comp=0,
          flags=[CAN_ELIMINATE, CAN_REORDER])
intrinsic("interp_deref_at_offset", src_comp=[1, 2], dest_comp=0,
          flags=[CAN_ELIMINATE, CAN_REORDER])
intrinsic("interp_deref_at_vertex", src_comp=[1, 1], dest_comp=0,
          flags=[CAN_ELIMINATE, CAN_REORDER])

# Gets the length of an unsized array at the end of a buffer
intrinsic("deref_buffer_array_length", src_comp=[-1], dest_comp=1,
          indices=[ACCESS], flags=[CAN_ELIMINATE, CAN_REORDER])

# Ask the driver for the size of a given SSBO. It takes the buffer index
# as source.
intrinsic("get_ssbo_size", src_comp=[-1], dest_comp=1, bit_sizes=[32],
          indices=[ACCESS], flags=[CAN_ELIMINATE, CAN_REORDER])
intrinsic("get_ubo_size", src_comp=[-1], dest_comp=1,
          flags=[CAN_ELIMINATE, CAN_REORDER])

# Intrinsics which provide a run-time mode-check.  Unlike the compile-time
# mode checks, a pointer can only have exactly one mode at runtime.
intrinsic("deref_mode_is", src_comp=[-1], dest_comp=1,
          indices=[MEMORY_MODES], flags=[CAN_ELIMINATE, CAN_REORDER])
intrinsic("addr_mode_is", src_comp=[-1], dest_comp=1,
          indices=[MEMORY_MODES], flags=[CAN_ELIMINATE, CAN_REORDER])

intrinsic("is_sparse_texels_resident", dest_comp=1, src_comp=[1], bit_sizes=[1,32],
          flags=[CAN_ELIMINATE, CAN_REORDER])
# result code is resident only if both inputs are resident
intrinsic("sparse_residency_code_and", dest_comp=1, src_comp=[1, 1], bit_sizes=[32],
          flags=[CAN_ELIMINATE, CAN_REORDER])

# a barrier is an intrinsic with no inputs/outputs but which can't be moved
# around/optimized in general
def barrier(name):
    intrinsic(name)

barrier("discard")

# Demote fragment shader invocation to a helper invocation.  Any stores to
# memory after this instruction are suppressed and the fragment does not write
# outputs to the framebuffer.  Unlike discard, demote needs to ensure that
# derivatives will still work for invocations that were not demoted.
#
# As specified by SPV_EXT_demote_to_helper_invocation.
barrier("demote")
intrinsic("is_helper_invocation", dest_comp=1, flags=[CAN_ELIMINATE])

# SpvOpTerminateInvocation from SPIR-V.  Essentially a discard "for real".
barrier("terminate")

# A workgroup-level control barrier.  Any thread which hits this barrier will
# pause until all threads within the current workgroup have also hit the
# barrier.  For compute shaders, the workgroup is defined as the local group.
# For tessellation control shaders, the workgroup is defined as the current
# patch.  This intrinsic does not imply any sort of memory barrier.
barrier("control_barrier")

# Memory barrier with semantics analogous to the memoryBarrier() GLSL
# intrinsic.
barrier("memory_barrier")

# Control/Memory barrier with explicit scope.  Follows the semantics of SPIR-V
# OpMemoryBarrier and OpControlBarrier, used to implement Vulkan Memory Model.
# Storage that the barrier applies is represented using NIR variable modes.
# For an OpMemoryBarrier, set EXECUTION_SCOPE to NIR_SCOPE_NONE.
intrinsic("scoped_barrier",
          indices=[EXECUTION_SCOPE, MEMORY_SCOPE, MEMORY_SEMANTICS, MEMORY_MODES])

# Shader clock intrinsic with semantics analogous to the clock2x32ARB()
# GLSL intrinsic.
# The latter can be used as code motion barrier, which is currently not
# feasible with NIR.
intrinsic("shader_clock", dest_comp=2, bit_sizes=[32], flags=[CAN_ELIMINATE],
          indices=[MEMORY_SCOPE])

# Shader ballot intrinsics with semantics analogous to the
#
#    ballotARB()
#    readInvocationARB()
#    readFirstInvocationARB()
#
# GLSL functions from ARB_shader_ballot.
intrinsic("ballot", src_comp=[1], dest_comp=0, flags=[CAN_ELIMINATE])
intrinsic("read_invocation", src_comp=[0, 1], dest_comp=0, bit_sizes=src0, flags=[CAN_ELIMINATE])
intrinsic("read_first_invocation", src_comp=[0], dest_comp=0, bit_sizes=src0, flags=[CAN_ELIMINATE])

# Returns the value of the first source for the lane where the second source is
# true. The second source must be true for exactly one lane.
intrinsic("read_invocation_cond_ir3", src_comp=[0, 1], dest_comp=0, flags=[CAN_ELIMINATE])

# Additional SPIR-V ballot intrinsics
#
# These correspond to the SPIR-V opcodes
#
#    OpGroupNonUniformElect
#    OpSubgroupFirstInvocationKHR
intrinsic("elect", dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("first_invocation", dest_comp=1, bit_sizes=[32], flags=[CAN_ELIMINATE])
intrinsic("last_invocation", dest_comp=1, bit_sizes=[32], flags=[CAN_ELIMINATE])

# Memory barrier with semantics analogous to the compute shader
# groupMemoryBarrier(), memoryBarrierAtomicCounter(), memoryBarrierBuffer(),
# memoryBarrierImage() and memoryBarrierShared() GLSL intrinsics.
barrier("group_memory_barrier")
barrier("memory_barrier_atomic_counter")
barrier("memory_barrier_buffer")
barrier("memory_barrier_image")
barrier("memory_barrier_shared")
barrier("begin_invocation_interlock")
barrier("end_invocation_interlock")

# Memory barrier for synchronizing TCS patch outputs
barrier("memory_barrier_tcs_patch")

# A conditional discard/demote/terminate, with a single boolean source.
intrinsic("discard_if", src_comp=[1])
intrinsic("demote_if", src_comp=[1])
intrinsic("terminate_if", src_comp=[1])

# ARB_shader_group_vote intrinsics
intrinsic("vote_any", src_comp=[1], dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("vote_all", src_comp=[1], dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("vote_feq", src_comp=[0], dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("vote_ieq", src_comp=[0], dest_comp=1, flags=[CAN_ELIMINATE])

# Ballot ALU operations from SPIR-V.
#
# These operations work like their ALU counterparts except that the operate
# on a uvec4 which is treated as a 128bit integer.  Also, they are, in
# general, free to ignore any bits which are above the subgroup size.
intrinsic("ballot_bitfield_extract", src_comp=[4, 1], dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("ballot_bit_count_reduce", src_comp=[4], dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("ballot_bit_count_inclusive", src_comp=[4], dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("ballot_bit_count_exclusive", src_comp=[4], dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("ballot_find_lsb", src_comp=[4], dest_comp=1, flags=[CAN_ELIMINATE])
intrinsic("ballot_find_msb", src_comp=[4], dest_comp=1, flags=[CAN_ELIMINATE])

# Shuffle operations from SPIR-V.
intrinsic("shuffle", src_comp=[0, 1], dest_comp=0, bit_sizes=src0, flags=[CAN_ELIMINATE])
intrinsic("shuffle_xor", src_comp=[0, 1], dest_comp=0, bit_sizes=src0, flags=[CAN_ELIMINATE])
intrinsic("shuffle_up", src_comp=[0, 1], dest_comp=0, bit_sizes=src0, flags=[CAN_ELIMINATE])
intrinsic("shuffle_down", src_comp=[0, 1], dest_comp=0, bit_sizes=src0, flags=[CAN_ELIMINATE])

# Quad operations from SPIR-V.
intrinsic("quad_broadcast", src_comp=[0, 1], dest_comp=0, flags=[CAN_ELIMINATE])
intrinsic("quad_swap_horizontal", src_comp=[0], dest_comp=0, flags=[CAN_ELIMINATE])
intrinsic("quad_swap_vertical", src_comp=[0], dest_comp=0, flags=[CAN_ELIMINATE])
intrinsic("quad_swap_diagonal", src_comp=[0], dest_comp=0, flags=[CAN_ELIMINATE])

intrinsic("reduce", src_comp=[0], dest_comp=0, bit_sizes=src0,
          indices=[REDUCTION_OP, CLUSTER_SIZE], flags=[CAN_ELIMINATE])
intrinsic("inclusive_scan", src_comp=[0], dest_comp=0, bit_sizes=src0,
          indices=[REDUCTION_OP], flags=[CAN_ELIMINATE])
intrinsic("exclusive_scan", src_comp=[0], dest_comp=0, bit_sizes=src0,
          indices=[REDUCTION_OP], flags=[CAN_ELIMINATE])

# AMD shader ballot operations
intrinsic("quad_swizzle_amd", src_comp=[0], dest_comp=0, bit_sizes=src0,
          indices=[SWIZZLE_MASK], flags=[CAN_ELIMINATE])
intrinsic("masked_swizzle_amd", src_comp=[0], dest_comp=0, bit_sizes=src0,
          indices=[SWIZZLE_MASK], flags=[CAN_ELIMINATE])
intrinsic("write_invocation_amd", src_comp=[0, 0, 1], dest_comp=0, bit_sizes=src0,
          flags=[CAN_ELIMINATE])
# src = [ mask, addition ]
intrinsic("mbcnt_amd", src_comp=[1, 1], dest_comp=1, bit_sizes=[32], flags=[CAN_ELIMINATE])
# Compiled to v_perm_b32. src = [ in_bytes_hi, in_bytes_lo, selector ]
intrinsic("byte_permute_amd", src_comp=[1, 1, 1], dest_comp=1, bit_sizes=[32], flags=[CAN_ELIMINATE, CAN_REORDER])
# Compiled to v_permlane16_b32. src = [ value, lanesel_lo, lanesel_hi ]
intrinsic("lane_permute_16_amd", src_comp=[1, 1, 1], dest_comp=1, bit_sizes=[32], flags=[CAN_ELIMINATE])

# Basic Geometry Shader intrinsics.
#
# emit_vertex implements GLSL's EmitStreamVertex() built-in.  It takes a single
# index, which is the stream ID to write to.
#
# end_primitive implements GLSL's EndPrimitive() built-in.
intrinsic("emit_vertex",   indices=[STREAM_ID])
intrinsic("end_primitive", indices=[STREAM_ID])

# Geometry Shader intrinsics with a vertex count.
#
# Alternatively, drivers may implement these intrinsics, and use
# nir_lower_gs_intrinsics() to convert from the basic intrinsics.
#
# These contain two additional unsigned integer sources:
# 1. The total number of vertices emitted so far.
# 2. The number of vertices emitted for the current primitive
#    so far if we're counting, otherwise undef.
intrinsic("emit_vertex_with_counter", src_comp=[1, 1], indices=[STREAM_ID])
intrinsic("end_primitive_with_counter", src_comp=[1, 1], indices=[STREAM_ID])
# Contains the final total vertex and primitive counts in the current GS thread.
intrinsic("set_vertex_and_primitive_count", src_comp=[1, 1], indices=[STREAM_ID])

# Launches mesh shader workgroups from a task shader, with explicit task_payload.
# Rules:
# - This is a terminating instruction.
# - May only occur in workgroup-uniform control flow.
# - Dispatch sizes may be divergent (in which case the values
#   from the first invocation are used).
# Meaning of indices:
# - BASE: address of the task_payload variable used.
# - RANGE: size of the task_payload variable used.
#
# src[] = {vec(x, y, z)}
intrinsic("launch_mesh_workgroups", src_comp=[3], indices=[BASE, RANGE])

# Launches mesh shader workgroups from a task shader, with task_payload variable deref.
# Same rules as launch_mesh_workgroups apply here as well.
# src[] = {vec(x, y, z), payload pointer}
intrinsic("launch_mesh_workgroups_with_payload_deref", src_comp=[3, -1], indices=[])

# Trace a ray through an acceleration structure
#
# This instruction has a lot of parameters:
#   0. Acceleration Structure
#   1. Ray Flags
#   2. Cull Mask
#   3. SBT Offset
#   4. SBT Stride
#   5. Miss shader index
#   6. Ray Origin
#   7. Ray Tmin
#   8. Ray Direction
#   9. Ray Tmax
#   10. Payload
intrinsic("trace_ray", src_comp=[-1, 1, 1, 1, 1, 1, 3, 1, 3, 1, -1])
# src[] = { hit_t, hit_kind }
intrinsic("report_ray_intersection", src_comp=[1, 1], dest_comp=1)
intrinsic("ignore_ray_intersection")
intrinsic("accept_ray_intersection") # Not in SPIR-V; useful for lowering
intrinsic("terminate_ray")
# src[] = { sbt_index, payload }
intrinsic("execute_callable", src_comp=[1, -1])

# Initialize a ray query
#
#   0. Ray Query
#   1. Acceleration Structure
#   2. Ray Flags
#   3. Cull Mask
#   4. Ray Origin
#   5. Ray Tmin
#   6. Ray Direction
#   7. Ray Tmax
intrinsic("rq_initialize", src_comp=[-1, -1, 1, 1, 3, 1, 3, 1])
# src[] = { query }
intrinsic("rq_terminate", src_comp=[-1])
# src[] = { query }
intrinsic("rq_proceed", src_comp=[-1], dest_comp=1)
# src[] = { query, hit }
intrinsic("rq_generate_intersection", src_comp=[-1, 1])
# src[] = { query }
intrinsic("rq_confirm_intersection", src_comp=[-1])
# src[] = { query, committed }
intrinsic("rq_load", src_comp=[-1, 1], dest_comp=0, indices=[RAY_QUERY_VALUE,COLUMN])

# Driver independent raytracing helpers

# rt_resume is a helper that that be the first instruction accesing the
# stack/scratch in a resume shader for a raytracing pipeline. It includes the
# resume index (for nir_lower_shader_calls_internal reasons) and the stack size
# of the variables spilled during the call. The stack size can be use to e.g.
# adjust a stack pointer.
intrinsic("rt_resume", indices=[CALL_IDX, STACK_SIZE])

# Lowered version of execute_callabe that includes the index of the resume
# shader, and the amount of scratch space needed for this call (.ie. how much
# to increase a stack pointer by).
# src[] = { sbt_index, payload }
intrinsic("rt_execute_callable", src_comp=[1, -1], indices=[CALL_IDX,STACK_SIZE])

# Lowered version of trace_ray in a similar vein to rt_execute_callable.
# src same as trace_ray
intrinsic("rt_trace_ray", src_comp=[-1, 1, 1, 1, 1, 1, 3, 1, 3, 1, -1],
          indices=[CALL_IDX, STACK_SIZE])


# Atomic counters
#
# The *_deref variants take an atomic_uint nir_variable, while the other,
# lowered, variants take a buffer index and register offset.  The buffer index
# is always constant, as there's no way to declare an array of atomic counter
# buffers.
#
# The register offset may be non-constant but must by dynamically uniform
# ("Atomic counters aggregated into arrays within a shader can only be indexed
# with dynamically uniform integral expressions, otherwise results are
# undefined.")
def atomic(name, flags=[]):
    intrinsic(name + "_deref", src_comp=[-1], dest_comp=1, flags=flags)
    intrinsic(name, src_comp=[1], dest_comp=1, indices=[BASE, RANGE_BASE], flags=flags)

def atomic2(name):
    intrinsic(name + "_deref", src_comp=[-1, 1], dest_comp=1)
    intrinsic(name, src_comp=[1, 1], dest_comp=1, indices=[BASE, RANGE_BASE])

def atomic3(name):
    intrinsic(name + "_deref", src_comp=[-1, 1, 1], dest_comp=1)
    intrinsic(name, src_comp=[1, 1, 1], dest_comp=1, indices=[BASE, RANGE_BASE])

atomic("atomic_counter_inc")
atomic("atomic_counter_pre_dec")
atomic("atomic_counter_post_dec")
atomic("atomic_counter_read", flags=[CAN_ELIMINATE])
atomic2("atomic_counter_add")
atomic2("atomic_counter_min")
atomic2("atomic_counter_max")
atomic2("atomic_counter_and")
atomic2("atomic_counter_or")
atomic2("atomic_counter_xor")
atomic2("atomic_counter_exchange")
atomic3("atomic_counter_comp_swap")

# Image load, store and atomic intrinsics.
#
# All image intrinsics come in three versions.  One which take an image target
# passed as a deref chain as the first source, one which takes an index as the
# first source, and one which takes a bindless handle as the first source.
# In the first version, the image variable contains the memory and layout
# qualifiers that influence the semantics of the intrinsic.  In the second and
# third, the image format and access qualifiers are provided as constant
# indices.  Up through GLSL ES 3.10, the image index source may only be a
# constant array access.  GLSL ES 3.20 and GLSL 4.00 allow dynamically uniform
# indexing.
#
# All image intrinsics take a four-coordinate vector and a sample index as
# 2nd and 3rd sources, determining the location within the image that will be
# accessed by the intrinsic.  Components not applicable to the image target
# in use are undefined.  Image store takes an additional four-component
# argument with the value to be written, and image atomic operations take
# either one or two additional scalar arguments with the same meaning as in
# the ARB_shader_image_load_store specification.
def image(name, src_comp=[], extra_indices=[], **kwargs):
    intrinsic("image_deref_" + name, src_comp=[-1] + src_comp,
              indices=[IMAGE_DIM, IMAGE_ARRAY, FORMAT, ACCESS] + extra_indices, **kwargs)
    intrinsic("image_" + name, src_comp=[1] + src_comp,
              indices=[IMAGE_DIM, IMAGE_ARRAY, FORMAT, ACCESS, RANGE_BASE] + extra_indices, **kwargs)
    intrinsic("bindless_image_" + name, src_comp=[-1] + src_comp,
              indices=[IMAGE_DIM, IMAGE_ARRAY, FORMAT, ACCESS] + extra_indices, **kwargs)

image("load", src_comp=[4, 1, 1], extra_indices=[DEST_TYPE], dest_comp=0, flags=[CAN_ELIMINATE])
image("sparse_load", src_comp=[4, 1, 1], extra_indices=[DEST_TYPE], dest_comp=0, flags=[CAN_ELIMINATE])
image("store", src_comp=[4, 1, 0, 1], extra_indices=[SRC_TYPE])
image("atomic_add",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_imin",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_umin",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_imax",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_umax",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_and",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_or",   src_comp=[4, 1, 1], dest_comp=1)
image("atomic_xor",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_exchange",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_comp_swap", src_comp=[4, 1, 1, 1], dest_comp=1)
image("atomic_fadd",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_fmin",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_fmax",  src_comp=[4, 1, 1], dest_comp=1)
image("size",    dest_comp=0, src_comp=[1], flags=[CAN_ELIMINATE, CAN_REORDER])
image("samples", dest_comp=1, flags=[CAN_ELIMINATE, CAN_REORDER])
image("atomic_inc_wrap",  src_comp=[4, 1, 1], dest_comp=1)
image("atomic_dec_wrap",  src_comp=[4, 1, 1], dest_comp=1)
# This returns true if all samples within the pixel have equal color values.
image("samples_identical", dest_comp=1, src_comp=[4], flags=[CAN_ELIMINATE])
# Non-uniform access is not lowered for image_descriptor_amd.
# dest_comp can be either 4 (buffer) or 8 (image).
image("descriptor_amd", dest_comp=0, src_comp=[], flags=[CAN_ELIMINATE, CAN_REORDER])
# CL-specific format queries
image("format", dest_comp=1, flags=[CAN_ELIMINATE, CAN_REORDER])
image("order", dest_comp=1, flags=[CAN_ELIMINATE, CAN_REORDER])
# Multisample fragment mask load
# src_comp[0] is same as image load src_comp[0]
image("fragment_mask_load_amd", src_comp=[4], dest_comp=1, bit_sizes=[32], flags=[CAN_ELIMINATE, CAN_REORDER])

# Vulkan descriptor set intrinsics
#
# The Vulkan API uses a different binding model from GL.  In the Vulkan
# API, all external resources are represented by a tuple:
#
# (descriptor set, binding, array index)
#
# where the array index is the only thing allowed to be indirect.  The
# vulkan_surface_index intrinsic takes the descriptor set and binding as
# its first two indices and the array index as its source.  The third
# index is a nir_variable_mode in case that's useful to the backend.
#
# The intended usage is that the shader will call vulkan_surface_index to
# get an index and then pass that as the buffer index ubo/ssbo calls.
#
# The vulkan_resource_reindex intrinsic takes a resource index in src0
# (the result of a vulkan_resource_index or vulkan_resource_reindex) which
# corresponds to the tuple (set, binding, index) and computes an index
# corresponding to tuple (set, binding, idx + src1).
intrinsic("vulkan_resource_index", src_comp=[1], dest_comp=0,
          indices=[DESC_SET, BINDING, DESC_TYPE],
          flags=[CAN_ELIMINATE, CAN_REORDER])
intrinsic("vulkan_resource_reindex", src_comp=[0, 1], dest_comp=0,
          indices=[DESC_TYPE], flags=[CAN_ELIMINATE, CAN_REORDER])
intrinsic("load_vulkan_descriptor", src_comp=[-1], dest_comp=0,
          indices=[DESC_TYPE], flags=[CAN_ELIMINATE, CAN_REORDER])

# atomic intrinsics
#
# All of these atomic memory operations read a value from memory, compute a new
# value using one of the operations below, write the new value to memory, and
# return the original value read.
#
# All variable operations take 2 sources except CompSwap that takes 3. These
# sources represent:
#
# 0: A deref to the memory on which to perform the atomic
# 1: The data parameter to the atomic function (i.e. the value to add
#    in shared_atomic_add, etc).
# 2: For CompSwap only: the second data parameter.
#
# All SSBO operations take 3 sources except CompSwap that takes 4. These
# sources represent:
#
# 0: The SSBO buffer index (dynamically uniform in GLSL, possibly non-uniform
#    with VK_EXT_descriptor_indexing).
# 1: The offset into the SSBO buffer of the variable that the atomic
#    operation will operate on.
# 2: The data parameter to the atomic function (i.e. the value to add
#    in ssbo_atomic_add, etc).
# 3: For CompSwap only: the second data parameter.
#
# All shared (and task payload) variable operations take 2 sources
# except CompSwap that takes 3.
# These sources represent:
#
# 0: The offset into the shared variable storage region that the atomic
#    operation will operate on.
# 1: The data parameter to the atomic function (i.e. the value to add
#    in shared_atomic_add, etc).
# 2: For CompSwap only: the second data parameter.
#
# All global operations take 2 sources except CompSwap that takes 3. These
# sources represent:
#
# 0: The memory address that the atomic operation will operate on.
# 1: The data parameter to the atomic function (i.e. the value to add
#    in shared_atomic_add, etc).
# 2: For CompSwap only: the second data parameter.
#
# The 2x32 global variants use a vec2 for the memory address where component X
# has the low 32-bit and component Y has the high 32-bit.
#
# IR3 global operations take 32b vec2 as memory address. IR3 doesn't support
# float atomics.

def memory_atomic_data1(name):
    intrinsic("deref_atomic_" + name,  src_comp=[-1, 1], dest_comp=1, indices=[ACCESS])
    intrinsic("ssbo_atomic_" + name,  src_comp=[-1, 1, 1], dest_comp=1, indices=[ACCESS])
    intrinsic("shared_atomic_" + name,  src_comp=[1, 1], dest_comp=1, indices=[BASE])
    intrinsic("task_payload_atomic_" + name,  src_comp=[1, 1], dest_comp=1, indices=[BASE])
    intrinsic("global_atomic_" + name,  src_comp=[1, 1], dest_comp=1, indices=[])
    intrinsic("global_atomic_" + name + "_2x32",  src_comp=[2, 1], dest_comp=1, indices=[])
    intrinsic("global_atomic_" + name + "_amd",  src_comp=[1, 1, 1], dest_comp=1, indices=[BASE])
    if not name.startswith('f'):
        intrinsic("global_atomic_" + name + "_ir3",  src_comp=[2, 1], dest_comp=1, indices=[BASE])

def memory_atomic_data2(name):
    intrinsic("deref_atomic_" + name,  src_comp=[-1, 1, 1], dest_comp=1, indices=[ACCESS])
    intrinsic("ssbo_atomic_" + name,  src_comp=[-1, 1, 1, 1], dest_comp=1, indices=[ACCESS])
    intrinsic("shared_atomic_" + name,  src_comp=[1, 1, 1], dest_comp=1, indices=[BASE])
    intrinsic("task_payload_atomic_" + name,  src_comp=[1, 1, 1], dest_comp=1, indices=[BASE])
    intrinsic("global_atomic_" + name,  src_comp=[1, 1, 1], dest_comp=1, indices=[])
    intrinsic("global_atomic_" + name + "_2x32",  src_comp=[2, 1, 1], dest_comp=1, indices=[])
    intrinsic("global_atomic_" + name + "_amd",  src_comp=[1, 1, 1, 1], dest_comp=1, indices=[BASE])
    if not name.startswith('f'):
        intrinsic("global_atomic_" + name + "_ir3",  src_comp=[2, 1, 1], dest_comp=1, indices=[BASE])

memory_atomic_data1("add")
memory_atomic_data1("imin")
memory_atomic_data1("umin")
memory_atomic_data1("imax")
memory_atomic_data1("umax")
memory_atomic_data1("and")
memory_atomic_data1("or")
memory_atomic_data1("xor")
memory_atomic_data1("exchange")
memory_atomic_data1("fadd")
memory_atomic_data1("fmin")
memory_atomic_data1("fmax")
memory_atomic_data2("comp_swap")
memory_atomic_data2("fcomp_swap")

def system_value(name, dest_comp, indices=[], bit_sizes=[32]):
    intrinsic("load_" + name, [], dest_comp, indices,
              flags=[CAN_ELIMINATE, CAN_REORDER], sysval=True,
              bit_sizes=bit_sizes)

system_value("frag_coord", 4)
system_value("point_coord", 2)
system_value("line_coord", 1)
system_value("front_face", 1, bit_sizes=[1, 32])
system_value("vertex_id", 1)
system_value("vertex_id_zero_base", 1)
system_value("first_vertex", 1)
system_value("is_indexed_draw", 1)
system_value("base_vertex", 1)
system_value("instance_id", 1)
system_value("base_instance", 1)
system_value("draw_id", 1)
system_value("sample_id", 1)
# sample_id_no_per_sample is like sample_id but does not imply per-
# sample shading.  See the lower_helper_invocation option.
system_value("sample_id_no_per_sample", 1)
system_value("sample_pos", 2)
# sample_pos_or_center is like sample_pos but does not imply per-sample
# shading.  When per-sample dispatch is not enabled, it returns (0.5, 0.5).
system_value("sample_pos_or_center", 2)
system_value("sample_mask_in", 1)
system_value("primitive_id", 1)
system_value("invocation_id", 1)
system_value("tess_coord", 3)
system_value("tess_level_outer", 4)
system_value("tess_level_inner", 2)
system_value("tess_level_outer_default", 4)
system_value("tess_level_inner_default", 2)
system_value("patch_vertices_in", 1)
system_value("local_invocation_id", 3)
system_value("local_invocation_index", 1)
# zero_base indicates it starts from 0 for the current dispatch
# non-zero_base indicates the base is included
system_value("workgroup_id", 3, bit_sizes=[32, 64])
system_value("workgroup_id_zero_base", 3)
# The workgroup_index is intended for situations when a 3 dimensional
# workgroup_id is not available on the HW, but a 1 dimensional index is.
system_value("workgroup_index", 1)
system_value("base_workgroup_id", 3, bit_sizes=[32, 64])
system_value("user_clip_plane", 4, indices=[UCP_ID])
system_value("num_workgroups", 3, bit_sizes=[32, 64])
system_value("num_vertices", 1)
system_value("helper_invocation", 1, bit_sizes=[1, 32])
system_value("layer_id", 1)
system_value("view_index", 1)
system_value("subgroup_size", 1)
system_value("subgroup_invocation", 1)
system_value("subgroup_eq_mask", 0, bit_sizes=[32, 64])
system_value("subgroup_ge_mask", 0, bit_sizes=[32, 64])
system_value("subgroup_gt_mask", 0, bit_sizes=[32, 64])
system_value("subgroup_le_mask", 0, bit_sizes=[32, 64])
system_value("subgroup_lt_mask", 0, bit_sizes=[32, 64])
system_value("num_subgroups", 1)
system_value("subgroup_id", 1)
system_value("workgroup_size", 3)
# note: the definition of global_invocation_id_zero_base is based on
# (workgroup_id * workgroup_size) + local_invocation_id.
# it is *not* based on workgroup_id_zero_base, meaning the work group
# base is already accounted for, and the global base is additive on top of that
system_value("global_invocation_id", 3, bit_sizes=[32, 64])
system_value("global_invocation_id_zero_base", 3, bit_sizes=[32, 64])
system_value("base_global_invocation_id", 3, bit_sizes=[32, 64])
system_value("global_invocation_index", 1, bit_sizes=[32, 64])
system_value("work_dim", 1)
system_value("line_width", 1)
system_value("aa_line_width", 1)
# BASE=0 for global/shader, BASE=1 for local/function
system_value("scratch_base_ptr", 0, bit_sizes=[32,64], indices=[BASE])
system_value("constant_base_ptr", 0, bit_sizes=[32,64])
system_value("shared_base_ptr", 0, bit_sizes=[32,64])
system_value("global_base_ptr", 0, bit_sizes=[32,64])
# Address of a transform feedback buffer, indexed by BASE
system_value("xfb_address", 1, bit_sizes=[32,64], indices=[BASE])

# System values for ray tracing.
system_value("ray_launch_id", 3)
system_value("ray_launch_size", 3)
system_value("ray_world_origin", 3)
system_value("ray_world_direction", 3)
system_value("ray_object_origin", 3)
system_value("ray_object_direction", 3)
system_value("ray_t_min", 1)
system_value("ray_t_max", 1)
system_value("ray_object_to_world", 3, indices=[COLUMN])
system_value("ray_world_to_object", 3, indices=[COLUMN])
system_value("ray_hit_kind", 1)
system_value("ray_flags", 1)
system_value("ray_geometry_index", 1)
system_value("ray_instance_custom_index", 1)
system_value("shader_record_ptr", 1, bit_sizes=[64])
system_value("cull_mask", 1)

# Driver-specific viewport scale/offset parameters.
#
# VC4 and V3D need to emit a scaled version of the position in the vertex
# shaders for binning, and having system values lets us move the math for that
# into NIR.
#
# Panfrost needs to implement all coordinate transformation in the
# vertex shader; system values allow us to share this routine in NIR.
system_value("viewport_x_scale", 1)
system_value("viewport_y_scale", 1)
system_value("viewport_z_scale", 1)
system_value("viewport_x_offset", 1)
system_value("viewport_y_offset", 1)
system_value("viewport_z_offset", 1)
system_value("viewport_scale", 3)
system_value("viewport_offset", 3)
# Pack xy scale and offset into a vec4 load (used by AMD NGG primitive culling)
system_value("viewport_xy_scale_and_offset", 4)

# Blend constant color values.  Float values are clamped. Vectored versions are
# provided as well for driver convenience

system_value("blend_const_color_r_float", 1)
system_value("blend_const_color_g_float", 1)
system_value("blend_const_color_b_float", 1)
system_value("blend_const_color_a_float", 1)
system_value("blend_const_color_rgba", 4)
system_value("blend_const_color_rgba8888_unorm", 1)
system_value("blend_const_color_aaaa8888_unorm", 1)

# System values for gl_Color, for radeonsi which interpolates these in the
# shader prolog to handle two-sided color without recompiles and therefore
# doesn't handle these in the main shader part like normal varyings.
system_value("color0", 4)
system_value("color1", 4)

# System value for internal compute shaders in radeonsi.
system_value("user_data_amd", 4)

# Barycentric coordinate intrinsics.
#
# These set up the barycentric coordinates for a particular interpolation.
# The first four are for the simple cases: pixel, centroid, per-sample
# (at gl_SampleID), or pull model (1/W, 1/I, 1/J) at the pixel center. The next
# two handle interpolating at a specified sample location, or interpolating
# with a vec2 offset,
#
# The interp_mode index should be either the INTERP_MODE_SMOOTH or
# INTERP_MODE_NOPERSPECTIVE enum values.
#
# The vec2 value produced by these intrinsics is intended for use as the
# barycoord source of a load_interpolated_input intrinsic.

def barycentric(name, dst_comp, src_comp=[]):
    intrinsic("load_barycentric_" + name, src_comp=src_comp, dest_comp=dst_comp,
              indices=[INTERP_MODE], flags=[CAN_ELIMINATE, CAN_REORDER])

# no sources.
barycentric("pixel", 2)
barycentric("centroid", 2)
barycentric("sample", 2)
barycentric("model", 3)
# src[] = { sample_id }.
barycentric("at_sample", 2, [1])
# src[] = { offset.xy }.
barycentric("at_offset", 2, [2])

# Load sample position:
#
# Takes a sample # and returns a sample position.  Used for lowering
# interpolateAtSample() to interpolateAtOffset()
intrinsic("load_sample_pos_from_id", src_comp=[1], dest_comp=2,
          flags=[CAN_ELIMINATE, CAN_REORDER])

intrinsic("load_persp_center_rhw_ir3", dest_comp=1,
          flags=[CAN_ELIMINATE, CAN_REORDER])

# Load texture scaling values:
#
# Takes a sampler # and returns 1/size values for multiplying to normalize
# texture coordinates.  Used for lowering rect textures.
intrinsic("load_texture_rect_scaling", src_comp=[1], dest_comp=2,
          flags=[CAN_ELIMINATE, CAN_REORDER])

# Fragment shader input interpolation delta intrinsic.
#
# For hw where fragment shader input interpolation is handled in shader, the
# load_fs_input_interp deltas intrinsics can be used to load the input deltas
# used for interpolation as follows:
#
#    vec3 iid = load_fs_input_interp_deltas(varying_slot)
#    vec2 bary = load_barycentric_*(...)
#    float result = iid.x + iid.y * bary.y + iid.z * bary.x

intrinsic("load_fs_input_interp_deltas", src_comp=[1], dest_comp=3,
          indices=[BASE, COMPONENT, IO_SEMANTICS], flags=[CAN_ELIMINATE, CAN_REORDER])

# Load operations pull data from some piece of GPU memory.  All load
# operations operate in terms of offsets into some piece of theoretical
# memory.  Loads from externally visible memory (UBO and SSBO) simply take a
# byte offset as a source.  Loads from opaque memory (uniforms, inputs, etc.)
# take a base+offset pair where the nir_intrinsic_base() gives the location
# of the start of the variable being loaded and and the offset source is a
# offset into that variable.
#
# Uniform load operations have a nir_intrinsic_range() index that specifies the
# range (starting at base) of the data from which we are loading.  If
# range == 0, then the range is unknown.
#
# UBO load operations have a nir_intrinsic_range_base() and
# nir_intrinsic_range() that specify the byte range [range_base,
# range_base+range] of the UBO that the src offset access must lie within.
#
# Some load operations such as UBO/SSBO load and per_vertex loads take an
# additional source to specify which UBO/SSBO/vertex to load from.
#
# The exact address type depends on the lowering pass that generates the
# load/store intrinsics.  Typically, this is vec4 units for things such as
# varying slots and float units for fragment shader inputs.  UBO and SSBO
# offsets are always in bytes.

def load(name, src_comp, indices=[], flags=[]):
    intrinsic("load_" + name, src_comp, dest_comp=0, indices=indices,
              flags=flags)

# src[] = { offset }.
load("uniform", [1], [BASE, RANGE, DEST_TYPE], [CAN_ELIMINATE, CAN_REORDER])
# src[] = { buffer_index, offset }.
load("ubo", [-1, 1], [ACCESS, ALIGN_MUL, ALIGN_OFFSET, RANGE_BASE, RANGE], flags=[CAN_ELIMINATE, CAN_REORDER])
# src[] = { buffer_index, offset in vec4 units }.  base is also in vec4 units.
load("ubo_vec4", [-1, 1], [ACCESS, BASE, COMPONENT], flags=[CAN_ELIMINATE, CAN_REORDER])
# src[] = { offset }.
load("input", [1], [BASE, COMPONENT, DEST_TYPE, IO_SEMANTICS], [CAN_ELIMINATE, CAN_REORDER])
# src[] = { vertex_id, offset }.
load("input_vertex", [1, 1], [BASE, COMPONENT, DEST_TYPE, IO_SEMANTICS], [CAN_ELIMINATE, CAN_REORDER])
# src[] = { vertex, offset }.
load("per_vertex_input", [1, 1], [BASE, COMPONENT, DEST_TYPE, IO_SEMANTICS], [CAN_ELIMINATE, CAN_REORDER])
# src[] = { barycoord, offset }.
load("interpolated_input", [2, 1], [BASE, COMPONENT, DEST_TYPE, IO_SEMANTICS], [CAN_ELIMINATE, CAN_REORDER])

# src[] = { buffer_index, offset }.
load("ssbo", [-1, 1], [ACCESS, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])
# src[] = { buffer_index }
load("ssbo_address", [1], [], [CAN_ELIMINATE, CAN_REORDER])
# src[] = { offset }.
load("output", [1], [BASE, COMPONENT, DEST_TYPE, IO_SEMANTICS], flags=[CAN_ELIMINATE])
# src[] = { vertex, offset }.
load("per_vertex_output", [1, 1], [BASE, COMPONENT, DEST_TYPE, IO_SEMANTICS], [CAN_ELIMINATE])
# src[] = { primitive, offset }.
load("per_primitive_output", [1, 1], [BASE, COMPONENT, DEST_TYPE, IO_SEMANTICS], [CAN_ELIMINATE])
# src[] = { offset }.
load("shared", [1], [BASE, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])
# src[] = { offset }.
load("task_payload", [1], [BASE, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])
# src[] = { offset }.
load("push_constant", [1], [BASE, RANGE], [CAN_ELIMINATE, CAN_REORDER])
# src[] = { offset }.
load("constant", [1], [BASE, RANGE, ALIGN_MUL, ALIGN_OFFSET],
     [CAN_ELIMINATE, CAN_REORDER])
# src[] = { offset }.
load("constant_non_opt", [1], [BASE, RANGE, ALIGN_MUL, ALIGN_OFFSET],
     [CAN_ELIMINATE, CAN_REORDER])
# src[] = { address }.
load("global", [1], [ACCESS, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])
# src[] = { address }.
load("global_2x32", [2], [ACCESS, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])
# src[] = { address }.
load("global_constant", [1], [ACCESS, ALIGN_MUL, ALIGN_OFFSET],
     [CAN_ELIMINATE, CAN_REORDER])
# src[] = { base_address, offset }.
load("global_constant_offset", [1, 1], [ACCESS, ALIGN_MUL, ALIGN_OFFSET],
     [CAN_ELIMINATE, CAN_REORDER])
# src[] = { base_address, offset, bound }.
load("global_constant_bounded", [1, 1, 1], [ACCESS, ALIGN_MUL, ALIGN_OFFSET],
     [CAN_ELIMINATE, CAN_REORDER])
# src[] = { address }.
load("kernel_input", [1], [BASE, RANGE, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE, CAN_REORDER])
# src[] = { offset }.
load("scratch", [1], [ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])

# Stores work the same way as loads, except now the first source is the value
# to store and the second (and possibly third) source specify where to store
# the value.  SSBO and shared memory stores also have a
# nir_intrinsic_write_mask()

def store(name, srcs, indices=[], flags=[]):
    intrinsic("store_" + name, [0] + srcs, indices=indices, flags=flags)

# src[] = { value, offset }.
store("output", [1], [BASE, WRITE_MASK, COMPONENT, SRC_TYPE, IO_SEMANTICS, IO_XFB, IO_XFB2])
# src[] = { value, vertex, offset }.
store("per_vertex_output", [1, 1], [BASE, WRITE_MASK, COMPONENT, SRC_TYPE, IO_SEMANTICS])
# src[] = { value, primitive, offset }.
store("per_primitive_output", [1, 1], [BASE, WRITE_MASK, COMPONENT, SRC_TYPE, IO_SEMANTICS])
# src[] = { value, block_index, offset }
store("ssbo", [-1, 1], [WRITE_MASK, ACCESS, ALIGN_MUL, ALIGN_OFFSET])
# src[] = { value, offset }.
store("shared", [1], [BASE, WRITE_MASK, ALIGN_MUL, ALIGN_OFFSET])
# src[] = { value, offset }.
store("task_payload", [1], [BASE, WRITE_MASK, ALIGN_MUL, ALIGN_OFFSET])
# src[] = { value, address }.
store("global", [1], [WRITE_MASK, ACCESS, ALIGN_MUL, ALIGN_OFFSET])
# src[] = { value, address }.
store("global_2x32", [2], [WRITE_MASK, ACCESS, ALIGN_MUL, ALIGN_OFFSET])
# src[] = { value, offset }.
store("scratch", [1], [ALIGN_MUL, ALIGN_OFFSET, WRITE_MASK])

# Intrinsic to load/store from the call stack.
# BASE is the offset relative to the current position of the stack
# src[] = { }.
intrinsic("load_stack", [], dest_comp=0,
          indices=[BASE, ALIGN_MUL, ALIGN_OFFSET, CALL_IDX, VALUE_ID],
          flags=[CAN_ELIMINATE])
# src[] = { value }.
intrinsic("store_stack", [0],
          indices=[BASE, ALIGN_MUL, ALIGN_OFFSET, WRITE_MASK, CALL_IDX, VALUE_ID])


# A bit field to implement SPIRV FragmentShadingRateKHR
# bit | name              | description
#   0 | Vertical2Pixels   | Fragment invocation covers 2 pixels vertically
#   1 | Vertical4Pixels   | Fragment invocation covers 4 pixels vertically
#   2 | Horizontal2Pixels | Fragment invocation covers 2 pixels horizontally
#   3 | Horizontal4Pixels | Fragment invocation covers 4 pixels horizontally
intrinsic("load_frag_shading_rate", dest_comp=1, bit_sizes=[32],
          flags=[CAN_ELIMINATE, CAN_REORDER])

# OpenCL printf instruction
# First source is a deref to the format string
# Second source is a deref to a struct containing the args
# Dest is success or failure
intrinsic("printf", src_comp=[1, 1], dest_comp=1, bit_sizes=[32])
# Since most drivers will want to lower to just dumping args
# in a buffer, nir_lower_printf will do that, but requires
# the driver to at least provide a base location
system_value("printf_buffer_address", 1, bit_sizes=[32,64])

# Mesh shading MultiView intrinsics
system_value("mesh_view_count", 1)
load("mesh_view_indices", [1], [BASE, RANGE], [CAN_ELIMINATE, CAN_REORDER])

# Used to pass values from the preamble to the main shader.
# This should use something similar to Vulkan push constants and load_preamble
# should be relatively cheap.
# For now we only support accesses with a constant offset.
load("preamble", [], indices=[BASE], flags=[CAN_ELIMINATE, CAN_REORDER])
store("preamble", [], indices=[BASE])

# IR3-specific version of most SSBO intrinsics. The only different
# compare to the originals is that they add an extra source to hold
# the dword-offset, which is needed by the backend code apart from
# the byte-offset already provided by NIR in one of the sources.
#
# NIR lowering pass 'ir3_nir_lower_io_offset' will replace the
# original SSBO intrinsics by these, placing the computed
# dword-offset always in the last source.
#
# The float versions are not handled because those are not supported
# by the backend.
store("ssbo_ir3", [1, 1, 1],
      indices=[WRITE_MASK, ACCESS, ALIGN_MUL, ALIGN_OFFSET])
load("ssbo_ir3",  [1, 1, 1],
     indices=[ACCESS, ALIGN_MUL, ALIGN_OFFSET], flags=[CAN_ELIMINATE])
intrinsic("ssbo_atomic_add_ir3",        src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_imin_ir3",       src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_umin_ir3",       src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_imax_ir3",       src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_umax_ir3",       src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_and_ir3",        src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_or_ir3",         src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_xor_ir3",        src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_exchange_ir3",   src_comp=[1, 1, 1, 1],    dest_comp=1, indices=[ACCESS])
intrinsic("ssbo_atomic_comp_swap_ir3",  src_comp=[1, 1, 1, 1, 1], dest_comp=1, indices=[ACCESS])

# System values for freedreno geometry shaders.
system_value("vs_primitive_stride_ir3", 1)
system_value("vs_vertex_stride_ir3", 1)
system_value("gs_header_ir3", 1)
system_value("primitive_location_ir3", 1, indices=[DRIVER_LOCATION])

# System values for freedreno tessellation shaders.
system_value("hs_patch_stride_ir3", 1)
system_value("tess_factor_base_ir3", 2)
system_value("tess_param_base_ir3", 2)
system_value("tcs_header_ir3", 1)
system_value("rel_patch_id_ir3", 1)

# System values for freedreno compute shaders.
system_value("subgroup_id_shift_ir3", 1)

# IR3-specific intrinsics for tessellation control shaders.  cond_end_ir3 end
# the shader when src0 is false and is used to narrow down the TCS shader to
# just thread 0 before writing out tessellation levels.
intrinsic("cond_end_ir3", src_comp=[1])
# end_patch_ir3 is used just before thread 0 exist the TCS and presumably
# signals the TE that the patch is complete and can be tessellated.
intrinsic("end_patch_ir3")

# IR3-specific load/store intrinsics. These access a buffer used to pass data
# between geometry stages - perhaps it's explicit access to the vertex cache.

# src[] = { value, offset }.
store("shared_ir3", [1], [BASE, ALIGN_MUL, ALIGN_OFFSET])
# src[] = { offset }.
load("shared_ir3", [1], [BASE, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])

# IR3-specific load/store global intrinsics. They take a 64-bit base address
# and a 32-bit offset.  The hardware will add the base and the offset, which
# saves us from doing 64-bit math on the base address.

# src[] = { value, address(vec2 of hi+lo uint32_t), offset }.
# const_index[] = { write_mask, align_mul, align_offset }
store("global_ir3", [2, 1], indices=[ACCESS, ALIGN_MUL, ALIGN_OFFSET])
# src[] = { address(vec2 of hi+lo uint32_t), offset }.
# const_index[] = { access, align_mul, align_offset }
load("global_ir3", [2, 1], indices=[ACCESS, ALIGN_MUL, ALIGN_OFFSET], flags=[CAN_ELIMINATE])

# IR3-specific bindless handle specifier. Similar to vulkan_resource_index, but
# without the binding because the hardware expects a single flattened index
# rather than a (binding, index) pair. We may also want to use this with GL.
# Note that this doesn't actually turn into a HW instruction.
intrinsic("bindless_resource_ir3", [1], dest_comp=1, indices=[DESC_SET], flags=[CAN_ELIMINATE, CAN_REORDER])

# IR3-specific intrinsics for shader preamble. These are meant to be used like
# this:
#
# if (preamble_start()) {
#    if (subgroupElect()) {
#       // preamble
#       ...
#       preamble_end();
#    }
# }
# // main shader
# ...

intrinsic("preamble_start_ir3", [], dest_comp=1, flags=[CAN_ELIMINATE, CAN_REORDER])

barrier("preamble_end_ir3")

# IR3-specific intrinsic for stc. Should be used in the shader preamble.
store("uniform_ir3", [], indices=[BASE])

# IR3-specific intrinsic for ldc.k. Copies UBO to constant file.
# base is the const file base in components, range is the amount to copy in
# vec4's.
intrinsic("copy_ubo_to_uniform_ir3", [1, 1], indices=[BASE, RANGE])

# DXIL specific intrinsics
# src[] = { value, mask, index, offset }.
intrinsic("store_ssbo_masked_dxil", [1, 1, 1, 1])
# src[] = { value, index }.
intrinsic("store_shared_dxil", [1, 1])
# src[] = { value, mask, index }.
intrinsic("store_shared_masked_dxil", [1, 1, 1])
# src[] = { value, index }.
intrinsic("store_scratch_dxil", [1, 1])
# src[] = { index }.
load("shared_dxil", [1], [], [CAN_ELIMINATE])
# src[] = { index }.
load("scratch_dxil", [1], [], [CAN_ELIMINATE])
# src[] = { deref_var, offset }
load("ptr_dxil", [1, 1], [], [])
# src[] = { index, 16-byte-based-offset }
load("ubo_dxil", [1, 1], [], [CAN_ELIMINATE, CAN_REORDER])

# DXIL Shared atomic intrinsics
#
# All of the shared variable atomic memory operations read a value from
# memory, compute a new value using one of the operations below, write the
# new value to memory, and return the original value read.
#
# All operations take 2 sources:
#
# 0: The index in the i32 array for by the shared memory region
# 1: The data parameter to the atomic function (i.e. the value to add
#    in shared_atomic_add, etc).
intrinsic("shared_atomic_add_dxil",  src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_imin_dxil", src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_umin_dxil", src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_imax_dxil", src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_umax_dxil", src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_and_dxil",  src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_or_dxil",   src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_xor_dxil",  src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_exchange_dxil", src_comp=[1, 1], dest_comp=1)
intrinsic("shared_atomic_comp_swap_dxil", src_comp=[1, 1, 1], dest_comp=1)

# Intrinsics used by the Midgard/Bifrost blend pipeline. These are defined
# within a blend shader to read/write the raw value from the tile buffer,
# without applying any format conversion in the process. If the shader needs
# usable pixel values, it must apply format conversions itself.
#
# These definitions are generic, but they are explicitly vendored to prevent
# other drivers from using them, as their semantics is defined in terms of the
# Midgard/Bifrost hardware tile buffer and may not line up with anything sane.
# One notable divergence is sRGB, which is asymmetric: raw_input_pan requires
# an sRGB->linear conversion, but linear values should be written to
# raw_output_pan and the hardware handles linear->sRGB.

# src[] = { value }
store("raw_output_pan", [], [])
store("combined_output_pan", [1, 1, 1, 4], [IO_SEMANTICS, COMPONENT, SRC_TYPE, DEST_TYPE])
load("raw_output_pan", [1], [IO_SEMANTICS], [CAN_ELIMINATE, CAN_REORDER])

# Loads the sampler paramaters <min_lod, max_lod, lod_bias>
# src[] = { sampler_index }
load("sampler_lod_parameters_pan", [1], flags=[CAN_ELIMINATE, CAN_REORDER])

# Loads the sample position array on Bifrost, in a packed Arm-specific format
system_value("sample_positions_pan", 1, bit_sizes=[64])

# R600 specific instrincs
#
# location where the tesselation data is stored in LDS
system_value("tcs_in_param_base_r600", 4)
system_value("tcs_out_param_base_r600", 4)
system_value("tcs_rel_patch_id_r600", 1)
system_value("tcs_tess_factor_base_r600", 1)

# the tess coords come as xy only, z has to be calculated
system_value("tess_coord_r600", 2)

# load as many components as needed giving per-component addresses
intrinsic("load_local_shared_r600", src_comp=[0], dest_comp=0, indices = [], flags = [CAN_ELIMINATE])

store("local_shared_r600", [1], [WRITE_MASK])
store("tf_r600", [])

# AMD GCN/RDNA specific intrinsics

# This barrier is a hint that prevents moving the instruction that computes
# src after this barrier. It's a constraint for the instruction scheduler.
# Otherwise it's identical to a move instruction.
# On AMD, it also forces the src value to be stored in a VGPR.
intrinsic("optimization_barrier_vgpr_amd", dest_comp=0, src_comp=[0],
          flags=[CAN_ELIMINATE])

# src[] = { descriptor, vector byte offset, scalar byte offset, index offset }
# The index offset is multiplied by the stride in the descriptor. The vertex/scalar byte offsets
# are in bytes.
intrinsic("load_buffer_amd", src_comp=[4, 1, 1, 1], dest_comp=0, indices=[BASE, MEMORY_MODES, ACCESS], flags=[CAN_ELIMINATE])
# src[] = { store value, descriptor, vector byte offset, scalar byte offset, index offset }
intrinsic("store_buffer_amd", src_comp=[0, 4, 1, 1, 1], indices=[BASE, WRITE_MASK, MEMORY_MODES, ACCESS])

# src[] = { address, unsigned 32-bit offset }.
load("global_amd", [1, 1], indices=[BASE, ACCESS, ALIGN_MUL, ALIGN_OFFSET], flags=[CAN_ELIMINATE])
# src[] = { value, address, unsigned 32-bit offset }.
store("global_amd", [1, 1], indices=[BASE, ACCESS, ALIGN_MUL, ALIGN_OFFSET, WRITE_MASK])

# Same as shared_atomic_add, but with GDS. src[] = {store_val, gds_addr, m0}
intrinsic("gds_atomic_add_amd",  src_comp=[1, 1, 1], dest_comp=1, indices=[BASE])

# src[] = { descriptor, add_value }
intrinsic("buffer_atomic_add_amd", src_comp=[4, 1], dest_comp=1, indices=[BASE])

# src[] = { sample_id, num_samples }
intrinsic("load_sample_positions_amd", src_comp=[1, 1], dest_comp=2, flags=[CAN_ELIMINATE, CAN_REORDER])

# Descriptor where TCS outputs are stored for TES
system_value("ring_tess_offchip_amd", 4)
system_value("ring_tess_offchip_offset_amd", 1)
# Descriptor where TCS outputs are stored for the HW tessellator
system_value("ring_tess_factors_amd", 4)
system_value("ring_tess_factors_offset_amd", 1)
# Descriptor where ES outputs are stored for GS to read on GFX6-8
system_value("ring_esgs_amd", 4)
system_value("ring_es2gs_offset_amd", 1)
# Address of the task shader draw ring (used for VARYING_SLOT_TASK_COUNT)
system_value("ring_task_draw_amd", 4)
# Address of the task shader payload ring (used for all other outputs)
system_value("ring_task_payload_amd", 4)
# Address of the mesh shader scratch ring (used for excess mesh shader outputs)
system_value("ring_mesh_scratch_amd", 4)
system_value("ring_mesh_scratch_offset_amd", 1)
# Pointer into the draw and payload rings
system_value("task_ring_entry_amd", 1)
# Pointer into the draw and payload rings
system_value("task_ib_addr", 2)
system_value("task_ib_stride", 1)
# Descriptor where NGG attributes are stored on GFX11.
system_value("ring_attr_amd", 4)
system_value("ring_attr_offset_amd", 1)

# Number of patches processed by each TCS workgroup
system_value("tcs_num_patches_amd", 1)
# Relative tessellation patch ID within the current workgroup
system_value("tess_rel_patch_id_amd", 1)
# Vertex offsets used for GS per-vertex inputs
system_value("gs_vertex_offset_amd", 1, [BASE])
# Number of rasterization samples
system_value("rasterization_samples_amd", 1)

# Descriptor where GS outputs are stored for GS copy shader to read on GFX6-9
system_value("ring_gsvs_amd", 4, indices=[STREAM_ID])
# Write offset in gsvs ring for legacy GS shader
system_value("ring_gs2vs_offset_amd", 1)

# Streamout configuration
system_value("streamout_config_amd", 1)
# Position to write within the streamout buffers
system_value("streamout_write_index_amd", 1)
# Offset to write within a streamout buffer
system_value("streamout_offset_amd", 1, indices=[BASE])

# AMD merged shader intrinsics

# Whether the current invocation index in the subgroup is less than the source. The source must be
# subgroup uniform and bits 0-7 must be less than or equal to the wave size.
intrinsic("is_subgroup_invocation_lt_amd", src_comp=[1], dest_comp=1, bit_sizes=[1], flags=[CAN_ELIMINATE])

# AMD NGG intrinsics

# Number of initial input vertices in the current workgroup.
system_value("workgroup_num_input_vertices_amd", 1)
# Number of initial input primitives in the current workgroup.
system_value("workgroup_num_input_primitives_amd", 1)
# For NGG passthrough mode only. Pre-packed argument for export_primitive_amd.
system_value("packed_passthrough_primitive_amd", 1)
# Whether NGG should execute shader query for pipeline statistics.
system_value("pipeline_stat_query_enabled_amd", dest_comp=1, bit_sizes=[1])
# Whether NGG should execute shader query for primitive generated.
system_value("prim_gen_query_enabled_amd", dest_comp=1, bit_sizes=[1])
# Whether NGG should execute shader query for primitive streamouted.
system_value("prim_xfb_query_enabled_amd", dest_comp=1, bit_sizes=[1])
# Merged wave info. Bits 0-7 are the ES thread count, 8-15 are the GS thread count, 16-24 is the
# GS Wave ID, 24-27 is the wave index in the workgroup, and 28-31 is the workgroup size in waves.
system_value("merged_wave_info_amd", dest_comp=1)
# Whether the shader should clamp vertex color outputs to [0, 1].
system_value("clamp_vertex_color_amd", dest_comp=1, bit_sizes=[1])
# Whether the shader should cull front facing triangles.
intrinsic("load_cull_front_face_enabled_amd", dest_comp=1, bit_sizes=[1], flags=[CAN_ELIMINATE])
# Whether the shader should cull back facing triangles.
intrinsic("load_cull_back_face_enabled_amd", dest_comp=1, bit_sizes=[1], flags=[CAN_ELIMINATE])
# True if face culling should use CCW (false if CW).
intrinsic("load_cull_ccw_amd", dest_comp=1, bit_sizes=[1], flags=[CAN_ELIMINATE])
# Whether the shader should cull small primitives that are not visible in a pixel.
intrinsic("load_cull_small_primitives_enabled_amd", dest_comp=1, bit_sizes=[1], flags=[CAN_ELIMINATE])
# Whether any culling setting is enabled in the shader.
intrinsic("load_cull_any_enabled_amd", dest_comp=1, bit_sizes=[1], flags=[CAN_ELIMINATE])
# Small primitive culling precision
intrinsic("load_cull_small_prim_precision_amd", dest_comp=1, bit_sizes=[32], flags=[CAN_ELIMINATE, CAN_REORDER])
# Initial edge flags in a Vertex Shader, packed into the format the HW needs for primitive export.
intrinsic("load_initial_edgeflags_amd", src_comp=[], dest_comp=1, bit_sizes=[32], indices=[])
# Exports the current invocation's vertex. This is a placeholder where all vertex attribute export instructions should be emitted.
intrinsic("export_vertex_amd", src_comp=[], indices=[])
# Exports the current invocation's primitive. src[] = {packed_primitive_data}.
intrinsic("export_primitive_amd", src_comp=[1], indices=[])
# Allocates export space for vertices and primitives. src[] = {num_vertices, num_primitives}.
intrinsic("alloc_vertices_and_primitives_amd", src_comp=[1, 1], indices=[])
# Overwrites VS input registers, for use with vertex compaction after culling. src = {vertex_id, instance_id}.
intrinsic("overwrite_vs_arguments_amd", src_comp=[1, 1], indices=[])
# Overwrites TES input registers, for use with vertex compaction after culling. src = {tes_u, tes_v, rel_patch_id, patch_id}.
intrinsic("overwrite_tes_arguments_amd", src_comp=[1, 1, 1, 1], indices=[])

# The address of the sbt descriptors.
system_value("sbt_base_amd", 1, bit_sizes=[64])

# 1. HW descriptor
# 2. BVH node(64-bit pointer as 2x32 ...)
# 3. ray extent
# 4. ray origin
# 5. ray direction
# 6. inverse ray direction (componentwise 1.0/ray direction)
intrinsic("bvh64_intersect_ray_amd", [4, 2, 1, 3, 3, 3], 4, flags=[CAN_ELIMINATE, CAN_REORDER])

# Return of a callable in raytracing pipelines
intrinsic("rt_return_amd")

# offset into scratch for the input callable data in a raytracing pipeline.
system_value("rt_arg_scratch_offset_amd", 1)

# Whether to call the anyhit shader for an intersection in an intersection shader.
system_value("intersection_opaque_amd", 1, bit_sizes=[1])

# Used for indirect ray tracing.
system_value("ray_launch_size_addr_amd", 1, bit_sizes=[64])

# Scratch base of callable stack for ray tracing.
system_value("rt_dynamic_callable_stack_base_amd", 1)

# Ray Tracing Traversal inputs
system_value("sbt_offset_amd", 1)
system_value("sbt_stride_amd", 1)
system_value("accel_struct_amd", 1, bit_sizes=[64])

#   0. SBT Index
#   1. Ray Tmax
#   2. Primitive Id
#   3. Instance Addr
#   4. Geometry Id and Flags
#   5. Hit Kind
intrinsic("execute_closest_hit_amd", src_comp=[1, 1, 1, 1, 1, 1])

#   0. Ray Tmax
intrinsic("execute_miss_amd", src_comp=[1])

# Used for saving and restoring hit attribute variables.
# BASE=dword index
intrinsic("load_hit_attrib_amd", dest_comp=1, bit_sizes=[32], indices=[BASE])
intrinsic("store_hit_attrib_amd", src_comp=[1], indices=[BASE])

# Load forced VRS rates.
intrinsic("load_force_vrs_rates_amd", dest_comp=1, bit_sizes=[32], flags=[CAN_ELIMINATE, CAN_REORDER])

intrinsic("load_scalar_arg_amd", dest_comp=0, bit_sizes=[32], indices=[BASE, ARG_UPPER_BOUND_U32_AMD], flags=[CAN_ELIMINATE, CAN_REORDER])
intrinsic("load_vector_arg_amd", dest_comp=0, bit_sizes=[32], indices=[BASE, ARG_UPPER_BOUND_U32_AMD], flags=[CAN_ELIMINATE, CAN_REORDER])

# src[] = { 32/64-bit base address, 32-bit offset }.
intrinsic("load_smem_amd", src_comp=[1, 1], dest_comp=0, bit_sizes=[32],
                           indices=[ALIGN_MUL, ALIGN_OFFSET],
                           flags=[CAN_ELIMINATE, CAN_REORDER])

# src[] = { descriptor, offset }
intrinsic("load_smem_buffer_amd", src_comp=[4, 1], dest_comp=0, bit_sizes=[32],
                                  indices=[ALIGN_MUL, ALIGN_OFFSET],
                                  flags=[CAN_ELIMINATE, CAN_REORDER])

# src[] = { offset }.
intrinsic("load_shared2_amd", [1], dest_comp=2, indices=[OFFSET0, OFFSET1, ST64], flags=[CAN_ELIMINATE])

# src[] = { value, offset }.
intrinsic("store_shared2_amd", [2, 1], indices=[OFFSET0, OFFSET1, ST64])

# Vertex stride in LS-HS buffer
system_value("lshs_vertex_stride_amd", 1)

# Per patch data offset in HS VRAM output buffer
system_value("hs_out_patch_data_offset_amd", 1)

# line_width * 0.5 / abs(viewport_scale[2])
system_value("clip_half_line_width_amd", 2)

# Number of vertices in a primitive
system_value("num_vertices_per_primitive_amd", 1)

# Load streamout buffer desc
# BASE = buffer index
intrinsic("load_streamout_buffer_amd", dest_comp=4, indices=[BASE], bit_sizes=[32], flags=[CAN_ELIMINATE, CAN_REORDER])

# An ID for each workgroup ordered by primitve sequence
system_value("ordered_id_amd", 1)

# Add to global streamout buffer counter in specified order
# src[] = { ordered_id, counter }
# WRITE_MASK = mask for counter channel to update
intrinsic("ordered_xfb_counter_add_amd", dest_comp=0, src_comp=[1, 0], indices=[WRITE_MASK], bit_sizes=[32])

# Provoking vertex index in a primitive
system_value("provoking_vtx_in_prim_amd", 1)

# Atomically add current wave's primitive count to query result
#   * GS emitted primitive is primitive emitted by any GS stream
#   * generated primitive is primitive that has been produced for that stream by VS/TES/GS
#   * streamout primitve is primitve that has been written to xfb buffer, may be different
#     than generated primitive when xfb buffer is too small to hold more primitives
# src[] = { primitive_count }.
intrinsic("atomic_add_gs_emit_prim_count_amd", [1])
intrinsic("atomic_add_gen_prim_count_amd", [1], indices=[STREAM_ID])
intrinsic("atomic_add_xfb_prim_count_amd", [1], indices=[STREAM_ID])

# Atomically add current wave's invocation count to query result
# src[] = { invocation_count }.
intrinsic("atomic_add_gs_invocation_count_amd", [1])

# LDS offset for scratch section in NGG shader
system_value("lds_ngg_scratch_base_amd", 1)
# LDS offset for NGG GS shader vertex emit
system_value("lds_ngg_gs_out_vertex_base_amd", 1)

# V3D-specific instrinc for tile buffer color reads.
#
# The hardware requires that we read the samples and components of a pixel
# in order, so we cannot eliminate or remove any loads in a sequence.
#
# src[] = { render_target }
# BASE = sample index
load("tlb_color_v3d", [1], [BASE, COMPONENT], [])

# V3D-specific instrinc for per-sample tile buffer color writes.
#
# The driver backend needs to identify per-sample color writes and emit
# specific code for them.
#
# src[] = { value, render_target }
# BASE = sample index
store("tlb_sample_color_v3d", [1], [BASE, COMPONENT, SRC_TYPE], [])

# V3D-specific intrinsic to load the number of layers attached to
# the target framebuffer
intrinsic("load_fb_layers_v3d", dest_comp=1, flags=[CAN_ELIMINATE, CAN_REORDER])

# Load/store a pixel in local memory. This operation is formatted, with
# conversion between the specified format and the implied register format of the
# source/destination (for store/loads respectively). This mostly matters for
# converting between floating-point registers and normalized memory formats.
#
# The format is the pipe_format of the local memory (the source), see
# agx_internal_formats.h for the supported list.
#
# Logically, this loads/stores a single sample. The sample to load is
# specified by the bitfield sample mask source. However, for stores multiple
# bits of the sample mask may be set, which will replicate the value. For
# pixel rate shading, use 0xFF as the mask to store to all samples regardless of
# the sample count.
#
# All calculations are relative to an immediate byte offset into local
# memory, which acts relative to the start of the sample. These instructions
# logically access:
#
#   (((((y * tile_width) + x) * nr_samples) + sample) * sample_stride) + offset
#
# src[] = { sample mask }
# base = offset
load("local_pixel_agx", [1], [BASE, FORMAT], [CAN_REORDER, CAN_ELIMINATE])
# src[] = { value, sample mask }
# base = offset
store("local_pixel_agx", [1], [BASE, WRITE_MASK, FORMAT], [CAN_REORDER])

# Combined depth/stencil emit, applying to a mask of samples. base indicates
# which to write (1 = depth, 2 = stencil, 3 = both).
#
# src[] = { sample mask, depth, stencil }
intrinsic("store_zs_agx", [1, 1, 1], indices=[BASE], flags=[])

# Store a block from local memory into a bound image. Used to write out render
# targets within the end-of-tile shader, although it is valid in general compute
# kernels.
#
# The format is the pipe_format of the local memory (the source), see
# agx_internal_formats.h for the supported list. The image format is
# specified in the PBE descriptor.
#
# The image dimension is used to distinguish multisampled images from
# non-multisampled images. It must be 2D or MS.
#
# src[] = { image index, logical offset within shared memory }
intrinsic("block_image_store_agx", [1, 1], bit_sizes=[32, 16],
          indices=[FORMAT, IMAGE_DIM], flags=[CAN_REORDER])

# Formatted load/store. The format is the pipe_format in memory (see
# agx_internal_formats.h for the supported list). This accesses:
#
#     address + extend(index) << (format shift + shift)
#
# The nir_intrinsic_base() index encodes the shift. The sign_extend index
# determines whether sign- or zero-extension is used for the index.
#
# All loads and stores on AGX uses these hardware instructions, so while these are
# logically load_global_agx/load_global_constant_agx/store_global_agx, the
# _global is omitted as it adds nothing.
#
# src[] = { address, index }.
load("agx", [1, 1], [ACCESS, BASE, FORMAT, SIGN_EXTEND], [CAN_ELIMINATE])
load("constant_agx", [1, 1], [ACCESS, BASE, FORMAT, SIGN_EXTEND],
     [CAN_ELIMINATE, CAN_REORDER])
# src[] = { value, address, index }.
store("agx", [1, 1], [ACCESS, BASE, FORMAT, SIGN_EXTEND])

# Logical complement of load_front_face, mapping to an AGX system value
system_value("back_face_agx", 1, bit_sizes=[1, 32])

# Loads the texture descriptor base for indexed (non-bindless) textures. On G13,
# the referenced array has stride 24.
system_value("texture_base_agx", 1, bit_sizes=[64])

# Load the base address of an indexed UBO/VBO (for lowering UBOs/VBOs)
intrinsic("load_ubo_base_agx", src_comp=[1], dest_comp=1, bit_sizes=[64],
          flags=[CAN_ELIMINATE, CAN_REORDER])
intrinsic("load_vbo_base_agx", src_comp=[1], dest_comp=1, bit_sizes=[64],
          flags=[CAN_ELIMINATE, CAN_REORDER])

# Intel-specific query for loading from the brw_image_param struct passed
# into the shader as a uniform.  The variable is a deref to the image
# variable. The const index specifies which of the six parameters to load.
intrinsic("image_deref_load_param_intel", src_comp=[1], dest_comp=0,
          indices=[BASE], flags=[CAN_ELIMINATE, CAN_REORDER])
image("load_raw_intel", src_comp=[1], dest_comp=0,
      flags=[CAN_ELIMINATE])
image("store_raw_intel", src_comp=[1, 0])

# Intrinsic to load a block of at least 32B of constant data from a 64-bit
# global memory address.  The memory address must be uniform and 32B-aligned.
# The second source is a predicate which indicates whether or not to actually
# do the load.
# src[] = { address, predicate }.
intrinsic("load_global_const_block_intel", src_comp=[1, 1], dest_comp=0,
          bit_sizes=[32], indices=[BASE], flags=[CAN_ELIMINATE, CAN_REORDER])

# Number of data items being operated on for a SIMD program.
system_value("simd_width_intel", 1)

# Load a relocatable 32-bit value
intrinsic("load_reloc_const_intel", dest_comp=1, bit_sizes=[32],
          indices=[PARAM_IDX], flags=[CAN_ELIMINATE, CAN_REORDER])

# 64-bit global address for a Vulkan descriptor set
# src[0] = { set }
intrinsic("load_desc_set_address_intel", dest_comp=1, bit_sizes=[64],
          src_comp=[1], flags=[CAN_ELIMINATE, CAN_REORDER])

# OpSubgroupBlockReadINTEL and OpSubgroupBlockWriteINTEL from SPV_INTEL_subgroups.
intrinsic("load_deref_block_intel", dest_comp=0, src_comp=[-1],
          indices=[ACCESS], flags=[CAN_ELIMINATE])
intrinsic("store_deref_block_intel", src_comp=[-1, 0], indices=[WRITE_MASK, ACCESS])

# src[] = { address }.
load("global_block_intel", [1], [ACCESS, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])

# src[] = { buffer_index, offset }.
load("ssbo_block_intel", [-1, 1], [ACCESS, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])

# src[] = { offset }.
load("shared_block_intel", [1], [BASE, ALIGN_MUL, ALIGN_OFFSET], [CAN_ELIMINATE])

# src[] = { value, address }.
store("global_block_intel", [1], [WRITE_MASK, ACCESS, ALIGN_MUL, ALIGN_OFFSET])

# src[] = { value, block_index, offset }
store("ssbo_block_intel", [-1, 1], [WRITE_MASK, ACCESS, ALIGN_MUL, ALIGN_OFFSET])

# src[] = { value, offset }.
store("shared_block_intel", [1], [BASE, WRITE_MASK, ALIGN_MUL, ALIGN_OFFSET])

# Intrinsics for Intel mesh shading
system_value("mesh_inline_data_intel", 1, [ALIGN_OFFSET], bit_sizes=[32, 64])

# Intrinsics for Intel bindless thread dispatch
# BASE=brw_topoloy_id
system_value("topology_id_intel", 1, indices=[BASE])
system_value("btd_stack_id_intel", 1)
system_value("btd_global_arg_addr_intel", 1, bit_sizes=[64])
system_value("btd_local_arg_addr_intel", 1, bit_sizes=[64])
system_value("btd_resume_sbt_addr_intel", 1, bit_sizes=[64])
# src[] = { global_arg_addr, btd_record }
intrinsic("btd_spawn_intel", src_comp=[1, 1])
# RANGE=stack_size
intrinsic("btd_stack_push_intel", indices=[STACK_SIZE])
# src[] = { }
intrinsic("btd_retire_intel")

# Intel-specific ray-tracing intrinsic
# src[] = { globals, level, operation } SYNCHRONOUS=synchronous
intrinsic("trace_ray_intel", src_comp=[1, 1, 1], indices=[SYNCHRONOUS])

# System values used for ray-tracing on Intel
system_value("ray_base_mem_addr_intel", 1, bit_sizes=[64])
system_value("ray_hw_stack_size_intel", 1)
system_value("ray_sw_stack_size_intel", 1)
system_value("ray_num_dss_rt_stacks_intel", 1)
system_value("ray_hit_sbt_addr_intel", 1, bit_sizes=[64])
system_value("ray_hit_sbt_stride_intel", 1, bit_sizes=[16])
system_value("ray_miss_sbt_addr_intel", 1, bit_sizes=[64])
system_value("ray_miss_sbt_stride_intel", 1, bit_sizes=[16])
system_value("callable_sbt_addr_intel", 1, bit_sizes=[64])
system_value("callable_sbt_stride_intel", 1, bit_sizes=[16])
system_value("leaf_opaque_intel", 1, bit_sizes=[1])
system_value("leaf_procedural_intel", 1, bit_sizes=[1])
# Values :
#  0: AnyHit
#  1: ClosestHit
#  2: Miss
#  3: Intersection
system_value("btd_shader_type_intel", 1)
system_value("ray_query_global_intel", 1, bit_sizes=[64])

# In order to deal with flipped render targets, gl_PointCoord may be flipped
# in the shader requiring a shader key or extra instructions or it may be
# flipped in hardware based on a state bit.  This version of gl_PointCoord
# is defined to be whatever thing the hardware can easily give you, so long as
# it's in normalized coordinates in the range [0, 1] across the point.
intrinsic("load_point_coord_maybe_flipped", dest_comp=2, bit_sizes=[32])
