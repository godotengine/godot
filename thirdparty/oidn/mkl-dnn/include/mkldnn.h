/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MKLDNN_H
#define MKLDNN_H

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/* All symbols shall be internal unless marked as MKLDNN_API */
#if defined _WIN32 || defined __CYGWIN__
#   define MKLDNN_HELPER_DLL_IMPORT __declspec(dllimport)
#   define MKLDNN_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#   if __GNUC__ >= 4
#       define MKLDNN_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
#       define MKLDNN_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
#   else
#       define MKLDNN_HELPER_DLL_IMPORT
#       define MKLDNN_HELPER_DLL_EXPORT
#   endif
#endif

#ifdef MKLDNN_DLL
#   ifdef MKLDNN_DLL_EXPORTS
#       define MKLDNN_API MKLDNN_HELPER_DLL_EXPORT
#   else
#       define MKLDNN_API MKLDNN_HELPER_DLL_IMPORT
#   endif
#else
#   define MKLDNN_API
#endif

#if defined (__GNUC__)
#   define MKLDNN_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#   define MKLDNN_DEPRECATED __declspec(deprecated)
#else
#   define MKLDNN_DEPRECATED
#endif

#include "mkldnn_types.h"
#include "mkldnn_version.h"
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup c_api C API
 * @{ */

/** @addtogroup c_api_primitive Primitive operations
 * @{ */

/** @addtogroup c_api_primitive_common Common primitive operations
 * @{ */

/** Creates a primitive descriptor @p iterator for given @p op_desc, @p attr,
 * @p engine, and optionally a hint primitive descriptor from forward
 * propagation (required for backward propagation). Pass @c NULL for forward
 * propagation.
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_iterator_create(
        mkldnn_primitive_desc_iterator_t *iterator,
        const_mkldnn_op_desc_t op_desc, const_mkldnn_primitive_attr_t attr,
        mkldnn_engine_t engine,
        const_mkldnn_primitive_desc_t hint_forward_primitive_desc);

/** Iterates over primitive descriptors. Returns #mkldnn_iterator_ends if no
 * more primitive descriptors are available. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_iterator_next(
        mkldnn_primitive_desc_iterator_t iterator);

/** Fetches the current primitive descriptor.
 *
 * @note
 *     The user should delete the fetched primitive descriptor using
 *     mkldnn_primitive_desc_destroy() once it is no longer needed. */
mkldnn_primitive_desc_t MKLDNN_API mkldnn_primitive_desc_iterator_fetch(
        const_mkldnn_primitive_desc_iterator_t iterator);

/** Deletes a primitive descriptor @p iterator */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_iterator_destroy(
        mkldnn_primitive_desc_iterator_t iterator);

/** Creates a @p primitive_desc using @p op_desc, @p attr, @p engine, and
 * optionally a hint primitive descriptor from forward propagation. The call is
 * equivalent to creating a primitive descriptor iterator, immediately fetching
 * a primitive descriptor, and then destroying the iterator. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_create(
        mkldnn_primitive_desc_t *primitive_desc,
        const_mkldnn_op_desc_t op_desc, const_mkldnn_primitive_attr_t attr,
        mkldnn_engine_t engine,
        const_mkldnn_primitive_desc_t hint_forward_primitive_desc);

/** Makes a copy of a @p primitive_desc. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_clone(
        mkldnn_primitive_desc_t *primitive_desc,
        const_mkldnn_primitive_desc_t existing_primitive_desc);

/** Returns a constant reference to the attribute of a @p primitive_desc.
 *
 * @warning
 *      The user should not destroy the obtained @p attr.
 *
 * @warning
 *      The lifetime of an @p attr is the same as that of a @p primitive_desc,
 *      so it is illegal to use the @p attr once @p primitive_desc has been
 *      destroyed. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_get_attr(
        const_mkldnn_primitive_desc_t primitive_desc,
        const_mkldnn_primitive_attr_t *attr);

/** Deletes a @p primitive_desc. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_destroy(
        mkldnn_primitive_desc_t primitive_desc);

/** Queries primitive descriptor
 *
 * One of the most typical use cases is to query a convolution primitive
 * descriptor created with source, weights, and destination formats equal
 * to #mkldnn_format_tag_any about the corresponding memory descriptors
 * (@p what equals #mkldnn_query_src_md, #mkldnn_query_weights_md, and
 * #mkldnn_query_dst_md respectively) to be able to prepare memory and
 * create reorders if required.
 *
 * Another quite typical use case is to query an operation primitive
 * descriptor for a workspace (@p what equals #mkldnn_query_workspace_md).
 * The returned status #mkldnn_not_required indicates that a workspace is
 * not required.
 *
 * A few other possibilities:
 *  - query an operation primitive descriptor for the underlying operation
 *    descriptor (#mkldnn_query_convolution_d, #mkldnn_query_eltwise_d,
 *    #mkldnn_query_rnn_d, etc.)
 *  - query an operation primitive descriptor for the implementation
 *    information string (#mkldnn_query_impl_info_str)
 *  - query an operation primitive descriptor for the number of inputs and
 *    outputs (#mkldnn_query_num_of_inputs_s32 and
 *    #mkldnn_query_num_of_outputs_s32 respectively)
 *
 * @sa mkldnn_query_t for more options
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_query(
        const_mkldnn_primitive_desc_t primitive_desc, mkldnn_query_t what,
        int index, void *result);

/** Queries primitive descriptor for memory descriptor
 *
 * @returns NULL in case of any error.
 *
 * This is just a specialized version of mkldnn_primitive_desc_query
 * used for convenience.
 */
const mkldnn_memory_desc_t MKLDNN_API *mkldnn_primitive_desc_query_md(
        const_mkldnn_primitive_desc_t primitive_desc, mkldnn_query_t what,
        int index);

/** Queries primitive descriptor for signed 32bit int
 *
 * @returns 0 in case of any error (in particular if the queried entity is
 * not of type int32_t). Note that 0 might also be the actual returned
 * value.
 *
 * This is just a specialized version of mkldnn_primitive_desc_query
 * used for convenience.
 */
int MKLDNN_API mkldnn_primitive_desc_query_s32(
        const_mkldnn_primitive_desc_t primitive_desc, mkldnn_query_t what,
        int index);

/** Creates a @p primitive using a @p primitive_desc descriptor. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_create(
        mkldnn_primitive_t *primitive,
        const_mkldnn_primitive_desc_t primitive_desc);

/** Executes a @p primitive using a @p stream, and @p nargs arguments
 * @p args. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_execute(
        const_mkldnn_primitive_t primitive, mkldnn_stream_t stream,
        int nargs, const mkldnn_exec_arg_t *args);

/** Retrieves a reference to the @p primitive_desc descriptor of given @p
 * primitive.
 *
 * @warning
 *     The returned object must not be destroyed by the user. The @c const
 *     qualifier of the returned object prevents such attempts. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_get_primitive_desc(
        const_mkldnn_primitive_t primitive,
        const_mkldnn_primitive_desc_t *primitive_desc);

/** Deletes a @p primitive. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_destroy(
        mkldnn_primitive_t primitive);

/** @} */

/** @addtogroup c_api_attributes Attributes
 * An extension for controlling primitive behavior.
 * @{ */

/** Creates an empty (default) @p attr attribute. All the parameters are set to
 * default values.
 *
 * An empty attribute is used in primitive descriptor creation whenever it
 * is not passed explicitly, e.g. in mkldnn_primitive_desc_create.
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_create(
        mkldnn_primitive_attr_t *attr);

/** Makes a copy of an @p existing_attr. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_clone(
        mkldnn_primitive_attr_t *attr,
        const_mkldnn_primitive_attr_t existing_attr);

/** Deletes an @p attr. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_destroy(
        mkldnn_primitive_attr_t attr);

/** Returns the scratchpad @p mode set in the attribute @p attr */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_get_scratchpad_mode(
        const_mkldnn_primitive_attr_t attr, mkldnn_scratchpad_mode_t *mode);

/** Sets scratchpad @p mode.
 *
 * The possible values are: #mkldnn_scratchpad_mode_library (default) and
 * #mkldnn_scratchpad_mode_user. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_set_scratchpad_mode(
        mkldnn_primitive_attr_t attr, mkldnn_scratchpad_mode_t mode);

/** Returns @p count, correspondence scale @p mask, and a pointer to a constant
 * floating point array of output @p scales for given @p attr, previously set
 * by mkldnn_primitive_attr_set_output_scales.
 *
 * @warning
 *      The @p scales array points to the internal @p attr field, so the user
 *      should not modify or destroy @p scales.
 *
 * @warning
 *      The lifetime of @p scales is the same as that of the @p attr to which it
 *      belongs, so it is illegal to use @p scales after @p attr is destroyed.
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_get_output_scales(
        const_mkldnn_primitive_attr_t attr, mkldnn_dim_t *count, int *mask,
        const float **scales);

/** Sets output @p scales for primitive operations. The number of elements @p
 * count and correspondence scale @p mask are stored for future use.
 *
 * The @p mask argument defines the correspondence between the output tensor
 * dimensions and the @p scales array. Set the i-th bit of @p mask to 1 to use a
 * dedicated scaling factor for each slice of the output tensor over the i-th
 * dimension. Set @p mask to 0 to use a common scaling factor for the whole
 * output tensor.
 *
 * @note
 *      The dimension order is always native and does not depend on the actual
 *      layout used. Examples:
 *       - 2D dimensional data the order of dimensions is always: (n, c)
 *       - 4D dimensional data the order is always: (n, c, h, w)
 *       - 5D dimensional weights the order is always: (g, oc, ic, kh, kw)
 *
 * Example usage:
 * @code
 *      int mb = 32, oc = 32, oh = 14, ow = 14; // convolution output params
 *      float scales[oc] = { ... }; // unique output scales per output channel
 *      int oc_dim = 1; // mb_dim = 0, channel_dim = 1, height_dim = 2, ...
 *
 *      mkldnn_convolution_desc_t cd; // create & configure convolution op_desc
 *
 *      mkldnn_primitive_attr_t attr;
 *      mkldnn_primitive_attr_create(&attr);  // create default attributes
 *      mkldnn_primitive_attr_set_output_scales(attr, oc, 1 << oc_dim, scales);
 *
 *      mkldnn_primitive_desc_t cpd;
 *      mkldnn_primitive_desc_create(&cpd, &cd, attr, NULL);
 * @endcode
 *
 * @note
 *      There is no way to check that @p count corresponds to @p mask until an
 *      actual primitive descriptor is created, so it is the user's
 *      responsibility to set proper values. The following formula must hold:
 *
 *      \f[count = \prod\limits_{d \in mask} output.dims[d]\f]
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_set_output_scales(
        mkldnn_primitive_attr_t attr, mkldnn_dim_t count, int mask,
        const float *scales);

/** Returns @p post_ops for given @p attr.
 *
 * @warning
 *      @p post_ops points to the internal @p attr field, so the user should not
 *      modify or destroy @p post_ops. Also, the lifetime of @p post_ops is the
 *      same as that of the @p attr it belongs to, so it is illegal to use @p
 *      post_ops after @p attr has been destroyed.
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_get_post_ops(
        const_mkldnn_primitive_attr_t attr, const_mkldnn_post_ops_t *post_ops);

/** Sets configured @p post_ops to an attribute @p attr for future use (when
 * primitive descriptor is being created).
 *
 * @note
 *      At this point in time, there is no way to check whether the primitive
 *      descriptor does or does not support a given sequence of post operations.
 *      Therefore the user should handle an error that might occur at the
 *      mkldnn_primitive_desc_create call.
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_set_post_ops(
        mkldnn_primitive_attr_t attr, const_mkldnn_post_ops_t post_ops);

/** @addtogroup c_api_attributes_post_ops Sequence of post operations
 * An extension for performing extra operations after a base operation.
 * @{ */

/** Creates an empty sequence of post operations @p post_ops. */
mkldnn_status_t MKLDNN_API mkldnn_post_ops_create(mkldnn_post_ops_t *post_ops);

/** Deletes a @p post_ops sequence. */
mkldnn_status_t MKLDNN_API mkldnn_post_ops_destroy(mkldnn_post_ops_t post_ops);

/** Returns the @p length of post operations for given @p post_ops. */
int MKLDNN_API mkldnn_post_ops_len(const_mkldnn_post_ops_t post_ops);

/** Returns the type of post operation with index @p index in given
 * @p post_ops. In case of error, returns #mkldnn_undefined_primitive. */
mkldnn_primitive_kind_t MKLDNN_API mkldnn_post_ops_get_kind(
        const_mkldnn_post_ops_t post_ops, int index);

/** Appends accumulation (sum) post operation to the @p post_ops. Prior to
 * accumulating the result, the previous value would be multiplied by @p scale.
 *
 * The kind of this post operation is #mkldnn_sum.
 *
 * This feature might improve performance for cases like residual learning
 * blocks, where the result of convolution is accumulated to the previously
 * computed activations. The parameter @p scale might be extreme for the
 * integer-based computations when the result and previous activations have
 * different logical scaling factors.
 *
 * In the simplest case when the accumulation is the only post operation, the
 * computations would be:
 * dst[] <- scale * dst[] + op(...) // instead of dst[] <- op(...)
 *
 * @note
 *      This post operation (as well as all the others) disregards the original
 *      layout of the destination; that is, the layout of the original
 *      destination is expected to be the same as the layout of the stored
 *      destination.
 */
mkldnn_status_t MKLDNN_API mkldnn_post_ops_append_sum(
        mkldnn_post_ops_t post_ops, float scale);

/** Gets the parameters of the accumulation (sum) post operation with index
 * @p index in the sequence of @p post_ops.
 *
 * @note
 *      If index @p index would not correspond to the accumulation post
 *      operation, the function returns #mkldnn_invalid_arguments.
 */
mkldnn_status_t MKLDNN_API mkldnn_post_ops_get_params_sum(
        const_mkldnn_post_ops_t post_ops, int index, float *scale);

/** Appends eltwise post operation to the @p post_ops with given parameters
 * @p kind, @p alpha, and @p beta (@sa mkldnn_eltwise_forward_desc_init and
 * mkldnn_eltwise_desc_t).
 *
 * The kind of this post operation is #mkldnn_eltwise.
 *
 * In the simplest case when the eltwise is the only post operation, the
 * computations would be:
 * dst[] <- scale * eltwise_op ( op(...) ) // instead of dst[] <- op(...)
 * where eltwise_op is configured with the given parameters.
 */
mkldnn_status_t MKLDNN_API mkldnn_post_ops_append_eltwise(
        mkldnn_post_ops_t post_ops, float scale, mkldnn_alg_kind_t alg,
        float alpha, float beta);

/** Gets the eltwise parameters of the post operation with index @p index in
 * the sequence of @p post_ops.
 */
mkldnn_status_t MKLDNN_API mkldnn_post_ops_get_params_eltwise(
        const_mkldnn_post_ops_t post_ops, int index, float *scale,
        mkldnn_alg_kind_t *alg, float *alpha, float *beta);

/** @} */

/** @} */

/** @addtogroup c_api_memory Memory
 * A primitive to describe and store data.
 *
 * The library supports various data types and formats. Memory hierarchy
 * consists of three levels of abstraction:
 * 1. **Memory descriptor** -- engine agnostic logical description of data
 *      (number of dimensions, dimensions themselves, and data type), and
 *      optionally the format/layout that describes the physical representation
 *      of data in memory. If the format is not known yet, one can pass
 *      #mkldnn_format_tag_any. This approach is used to allow compute-intensive
 *      primitives to specify the most appropriate format on their own with
 *      users required to reorder the data if the incoming format doesn't match
 *      the primitive's selection. Memory descriptor can be initialized with
 *      mkldnn_memory_desc_init_by_tag() or mkldnn_memory_desc_init_by_strides()
 *      functions, or by directly filling the mkldnn_memory_desc_t structure.
 *      The latter requires deep knowledge of how the physical data
 *      representation is mapped to the structure.
 *      The @ref understanding_memory_formats topic should shed some light on
 *      that.
 *      For the fully defined memory descriptors (i.e. where the format kind is
 *      not equal to #mkldnn_format_kind_any) a user can the size, using the
 *      mkldnn_memory_desc_get_size() function. As described in
 *      @ref understanding_memory_formats, the size of data sometimes cannot
 *      be computed as the product of dimensions times the size of the data
 *      type. So users are encouraged to use this function for better code
 *      portability.
 *      Two memory descriptors can be compared with mkldnn_memory_desc_equal().
 *      The comparison is especially useful when checking whether a primitive
 *      requires reorder from the user's data format to the primitive's format.
 * 2. **Memory** -- an engine-specific object that handles the data and its
 *      description (a memory descriptor). For CPU enigne, the data handle is
 *      simply a pointer to @c void. The data handle can be queried using
 *      mkldnn_memory_get_data_handle() and set using
 *      mkldnn_memory_set_data_handle(). The latter function always sets the
 *      memory in the padding region to zero, which is the invariant maintained
 *      by all the primitives in Intel MKL-DNN.
 *      See @ref understanding_memory_formats for more details.
 *      A memory can be created using mkldnn_memory_create() function.
 *      A memory can also be queried for the underlying memory descriptor and
 *      engine using mkldnn_memory_get_memory_desc() and
 *      mkldnn_memory_get_engine() functions.
 *
 * Along with ordinary memory with all dimensions being positive, Intel
 * MKL-DNN supports *zero-volume* memory with one or more dimensions set to
 * zero. This is to support the NumPy\* convention.
 * If a *zero-volume* memory is passed to a primitive, the primitive does
 * not perform any computations on this memory. For example:
 *  - Convolution with `(0 batch, 3 input channels, 13 height, 13 width)`
 *    source and `(16 output channels, 3 inputs, channel, 3 height, 3 width)`
 *    weights would produce `(0 batch, 16 output channels, 11 height, 11 width)`
 *    destination (assuming strides are `1` and paddings are zero) and perform
 *    zero multiply-add operations.
 *  - Concatenation of three memories of shapes `(3, 4, 13, 13)`,
 *    `(3, 0, 13, 13)`, and `(3, 1, 13, 13)` along the second axis would produce
 *    the output of the shape `(3, 5, 13, 13)`, effectively ignoring the second
 *    input (however, if the user created a concatenation primitive descriptor
 *    with three inputs they should also provide all three memories to the
 *    concatenation primitive, including the one with zero second dimension).
 *  - However, Intel MKL-DNN would return an error when attempting to create a
 *    convolution with *zero-volume* memory passed for weights because such a
 *    convolution is not well-defined:
 *    ~~~
 *    dst(1, 16, 11, 11) <-- src(1, 0, 13, 13) (*) wei(16, 0, 3, 3)
 *    ~~~
 *    Should the values in the destination be zeroes or just not accessed at
 *    all? Moreover, backward pass w.r.t. weights in such cases is also not
 *    well-defined.
 *
 *  Data handle of *zero-volume* memory is never accessed and hence can be
 *  unset (NULL in case of CPU engine).
 *
 * @sa @ref understanding_memory_formats
 * @{ */

/** Initializes a @p memory_desc memory descriptor using @p ndims, @p dims, @p
 * data_type, and @p strides.
 *
 * The @p strides might be NULL, which means the order of physical dimensions
 * is the same as the order of logical ones.
 *
 * @note The logical order of dimensions is defined by a primitive that
 *       consumes the memory.
 */
mkldnn_status_t MKLDNN_API mkldnn_memory_desc_init_by_strides(
        mkldnn_memory_desc_t *memory_desc, int ndims, const mkldnn_dims_t dims,
        mkldnn_data_type_t data_type, const mkldnn_dims_t strides);

/** Initializes a @p memory_desc memory descriptor using @p ndims, @p dims, @p
 * data_type, and format @p tag.
 *
 * @p tag can be #mkldnn_format_tag_any, which allows a primitive to define
 * the appropriate memory format. In this case, the @p format_kind would be set
 * to #mkldnn_format_kind_any */
mkldnn_status_t MKLDNN_API mkldnn_memory_desc_init_by_tag(
        mkldnn_memory_desc_t *memory_desc, int ndims, const mkldnn_dims_t dims,
        mkldnn_data_type_t data_type, mkldnn_format_tag_t tag);

/** Initializes a @p memory_desc for a given @p parent_memory_desc, with
 * @p dims sizes and @p offsets. May fail if layout used does not allow
 * obtain desired submemory. In this case consider using `extract` or `insert`
 * primitive */
mkldnn_status_t MKLDNN_API mkldnn_memory_desc_init_submemory(
        mkldnn_memory_desc_t *memory_desc,
        const mkldnn_memory_desc_t *parent_memory_desc,
        const mkldnn_dims_t dims, const mkldnn_dims_t offsets);

/** Compares two memory descriptors.
 * @return 1 if the descriptors are the same.
 * @return 0 if the descriptors are different.
 *
 * Use this function to identify whether a reorder is required between the
 * two memories */
int MKLDNN_API mkldnn_memory_desc_equal(
        const mkldnn_memory_desc_t *lhs,
        const mkldnn_memory_desc_t *rhs);

/** Returns the size (in bytes) that is required for given @p memory_desc */
size_t MKLDNN_API mkldnn_memory_desc_get_size(
        const mkldnn_memory_desc_t *memory_desc);

/** Creates a memory for given @p memory_desc and @p engine. Also sets handle
 * to @p native_handle.
 * The @p native_handle can:
 * - point to the user allocated memory, i.e. valid handle. In this case the
 *   library doesn't own allocated memory.
 * - be MKLDNN_NATIVE_HANDLE_ALLOCATE to ask the library to allocate and
 *   attach memory. In this case the library owns allocated memory.
 * - be MKLDNN_NATIVE_HANDLE_NONE to create mkldnn_memory w/o attached memory.
 */
mkldnn_status_t MKLDNN_API mkldnn_memory_create(mkldnn_memory_t *memory,
        const mkldnn_memory_desc_t *memory_desc, mkldnn_engine_t engine,
        void *native_handle);

/** Returns a @p memory_desc associated with @p memory. */
mkldnn_status_t MKLDNN_API mkldnn_memory_get_memory_desc(
        const_mkldnn_memory_t memory,
        const mkldnn_memory_desc_t **memory_desc);

/** Returns an @p engine associated with @p memory. */
mkldnn_status_t MKLDNN_API mkldnn_memory_get_engine(
        const_mkldnn_memory_t memory, mkldnn_engine_t *engine);

/** For a @p memory, returns the data @p handle.
 *
 * For the CPU engine, the data handle is a pointer to the actual data. */
mkldnn_status_t MKLDNN_API mkldnn_memory_get_data_handle(
        const_mkldnn_memory_t memory, void **handle);

/** For a @p memory, sets the data @p handle. */
mkldnn_status_t MKLDNN_API mkldnn_memory_set_data_handle(
        mkldnn_memory_t memory, void *handle);

/** Deletes a @p memory. */
mkldnn_status_t MKLDNN_API mkldnn_memory_destroy(mkldnn_memory_t memory);

/** @} */

/** @addtogroup c_api_reorder Reorder
 * A primitive to copy data between memory formats.
 * @{ */

/** Initializes a @p reorder_primitive_desc using the description of the source
 * (@p src_engine and @p src_md) and destination (@p dst_engine and @p dst_md)
 * memory, and an @p attr attribute.
 *
 * Inputs:
 *  - input (#mkldnn_query_src_md, 0)
 *
 * Outputs:
 *  - output (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_reorder_primitive_desc_create(
        mkldnn_primitive_desc_t *reorder_primitive_desc,
        mkldnn_engine_t src_engine, const mkldnn_memory_desc_t *src_md,
        mkldnn_engine_t dst_engine, const mkldnn_memory_desc_t *dst_md,
        const_mkldnn_primitive_attr_t attr);

/** @} */

/** @addtogroup c_api_concat Concat
 * A primitive to concatenate data by arbitrary dimension.
 * @{ */

/** Creates out-of-place @p concat_primitive_desc for concatenation of @p n
 * inputs by @p concat_dimension with resulting @p output_desc memory
 * descriptor. @p output_desc can be NULL or specified with the
 * #mkldnn_format_kind_any format kind -- in this case, the appropriate memory
 * format would be chosen automatically.
 *
 * Inputs:
 *  - input 0 (#mkldnn_query_src_md, 0)
 *  - input 1 (#mkldnn_query_src_md, 1)
 *  - ...
 *  - input @p n - 1 (#mkldnn_query_src_md, @p n - 1)
 *
 * Outputs:
 *  - output (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_concat_primitive_desc_create(
        mkldnn_primitive_desc_t *concat_primitive_desc,
        const mkldnn_memory_desc_t *dst_md,
        int n, int concat_dimension,
        const mkldnn_memory_desc_t *src_mds,
        const_mkldnn_primitive_attr_t attr,
        mkldnn_engine_t engine);

/** @} */

/** @addtogroup c_api_sum Sum
 * A primitive to sum data.
 * @{ */

/** Creates out-of-place @p sum_primitive_desc for sum of @p n
 * inputs multiplied by scale with resulting @p output_desc memory
 * descriptor. @p output_desc can be NULL or specified with the
 * #mkldnn_format_kind_any format kind -- in this case, the appropriate memory
 * format would be chosen automatically.
 *
 * Inputs:
 *  - src 0 (#mkldnn_query_src_md, 0)
 *  - src 1 (#mkldnn_query_src_md, 1)
 *  - ...
 *  - src @p n - 1 (#mkldnn_query_src_md, @p n - 1)
 *
 * Outputs:
 *  - output (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_sum_primitive_desc_create(
        mkldnn_primitive_desc_t *sum_primitive_desc,
        const mkldnn_memory_desc_t *dst_mds,
        int n, const float *scales,
        const mkldnn_memory_desc_t *src_mds,
        const_mkldnn_primitive_attr_t attr,
        mkldnn_engine_t engine);

/** @} */

/** @addtogroup c_api_convolution Convolution
 * A primitive to compute convolution using different algorithms.
 *
 * \f[dst[n][oc][oh][ow]  =
 *     \sum_{kw=0}^{KW}\sum_{kh=0}^{KH}\sum_{ic=0}^{IC}
 *     src[n][ic][oh \cdot s_h - p_l[0] + kh][ow \cdot s_w - p_r[1] + kw]
 *     \cdot weights[g][oc][ic][kh][kw]
 *     + bias[g][oc],\f]
 *
 * where size of output spatial domain is given by
 * \f$ OH = \left\lfloor{\frac{IH - KH + p_l[0] + p_r[0]}{s_h}}
 *          \right\rfloor + 1\f$,
 * \f$ OW = \left\lfloor{\frac{IW - KW + p_l[1] + p_r[1]}{s_w}}
 *          \right\rfloor + 1\f$,
 *
 * and summation is carried over input channels \f$ic\f$ in
 * group \f$g\f$, and \f$s_h, s_w\f$ are @p strides and
 * \f$p_l, p_r\f$ are @p padding_l and @p padding_r.
 * @{ */

/** Initializes a convolution descriptor @p conv_desc for forward propagation
 * using @p prop_kind (possible values are #mkldnn_forward_training and
 * #mkldnn_forward_inference), @p alg_kind, memory descriptors, @p strides, @p
 * padding_l, @p padding_r, and @p padding_kind. In order to create a
 * convolution without bias, @p bias_desc should either be @c NULL or point to
 * a descriptor with memory format kind equal to #mkldnn_format_kind_undef.
 *
 * @note If @p padding_r is @c NULL, the padding is supposed to be symmetric.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *  - bias (#mkldnn_query_weights_md, 1), if created with bias
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_convolution_forward_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a dilated convolution descriptor @p conv_desc for forward
 * propagation using @p prop_kind (possible values are #mkldnn_forward_training
 * and #mkldnn_forward_inference), @p alg_kind, memory descriptors, @p strides,
 * @p dilates, @p padding_l, @p padding_r, and @p padding_kind.
 * In order to create a dilated convolution without bias, @p bias_desc
 * should either be @c NULL or point to a descriptor with memory format kind
 * equals #mkldnn_format_kind_undef.
 *
 * @note If @p padding_r is @c NULL, the padding is supposed to be symmetric.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *  - bias (#mkldnn_query_weights_md, 1), if created with bias
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_dilated_convolution_forward_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t dilates, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** Initializes a convolution descriptor @p conv_desc for backward propagation
 * with respect to data using @p alg_kind, memory descriptors, @p strides, @p
 * padding_l, @p padding_r, and @p padding_kind.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_convolution_backward_data_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a dilated convolution descriptor @p conv_desc for backward
 * propagation with respect to data using @p alg_kind, memory descriptors, @p
 * strides, @p dilates @p padding_l, @p padding_r, and @p padding_kind.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_dilated_convolution_backward_data_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t dilates, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** Initializes a convolution descriptor @p conv_desc for backward propagation
 * with respect to weights using @p alg_kind, memory descriptors, @p strides,
 * @p padding_l, @p padding_r, and @p padding_kind.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *
 * Outputs:
 *  - diff_weights (#mkldnn_query_diff_weights_md, 0)
 *  - diff_bias (#mkldnn_query_diff_weights_md, 1), if created with bias
 */
mkldnn_status_t MKLDNN_API mkldnn_convolution_backward_weights_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *diff_weights_desc,
        const mkldnn_memory_desc_t *diff_bias_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a convolution descriptor @p conv_desc for backward propagation
 * with respect to weights using @p alg_kind, memory descriptors, @p strides,
 * @p dilates @p padding_l, @p padding_r, and @p padding_kind.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *
 * Outputs:
 *  - diff_weights (#mkldnn_query_diff_weights_md, 0)
 *  - diff_bias (#mkldnn_query_diff_weights_md, 1), if created with bias
 */
mkldnn_status_t MKLDNN_API
mkldnn_dilated_convolution_backward_weights_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *diff_weights_desc,
        const mkldnn_memory_desc_t *diff_bias_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t dilates, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** @} */

/** @addtogroup c_api_deconvolution Deconvolution
 * A primitive to compute deconvolution using different algorithms.
 *
 * @{ */


/** Initializes a deconvolution descriptor @p deconv_desc for forward
 * propagation using @p prop_kind (possible values are #mkldnn_forward_training
 * and #mkldnn_forward_inference), @p alg_kind, memory descriptors, @p strides,
 * @p padding_l, @p padding_r, and @p padding_kind. In order to create a
 * deconvolution without bias, @p bias_desc should either be @c NULL or point to
 * a descriptor with memory format kind equals #mkldnn_format_kind_undef.
 *
 * @note If @p padding_r is @c NULL, the padding is supposed to be symmetric.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *  - bias (#mkldnn_query_weights_md, 1), if created with bias
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_deconvolution_forward_desc_init(
        mkldnn_deconvolution_desc_t *conv_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a dilated deconvolution descriptor @p deconv_desc for forward
 * propagation using @p prop_kind (possible values are #mkldnn_forward_training
 * and #mkldnn_forward_inference), @p alg_kind, memory descriptors, @p strides,
 * @p dilates, @p padding_l, @p padding_r, and @p padding_kind. In order to
 * create a dilated deconvolution without bias, @p bias_desc should either be
 * @c NULL or point to a descriptor with memory format kind equal
 * #mkldnn_format_kind_undef.
 *
 * @note If @p padding_r is @c NULL, the padding is supposed to be symmetric.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *  - bias (#mkldnn_query_weights_md, 1), if created with bias
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_dilated_deconvolution_forward_desc_init(
        mkldnn_deconvolution_desc_t *conv_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t dilates, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** Initializes a deconvolution descriptor @p conv_desc for backward propagation
 * with respect to data using @p alg_kind, memory descriptors, @p strides, @p
 * padding_l, @p padding_r, and @p padding_kind.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_deconvolution_backward_data_desc_init(
        mkldnn_deconvolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a dilated deconvolution descriptor @p conv_desc for backward
 * propagation with respect to data using @p alg_kind, memory descriptors, @p
 * strides, @p dilates, @p padding_l, @p padding_r, and @p padding_kind.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_dilated_deconvolution_backward_data_desc_init(
        mkldnn_deconvolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t dilates, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** Initializes a deconvolution descriptor @p conv_desc for backward propagation
 * with respect to weights using @p alg_kind, memory descriptors, @p strides,
 * @p padding_l, @p padding_r, and @p padding_kind.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *
 * Outputs:
 *  - diff_weights (#mkldnn_query_diff_weights_md, 0)
 *  - diff_bias (#mkldnn_query_diff_weights_md, 1), if created with bias
 */
mkldnn_status_t MKLDNN_API mkldnn_deconvolution_backward_weights_desc_init(
        mkldnn_deconvolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *diff_weights_desc,
        const mkldnn_memory_desc_t *diff_bias_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a dilated deconvolution descriptor @p conv_desc for backward
 * propagation with respect to weights using @p alg_kind, memory descriptors,
 * @p strides, @p dilates, @p padding_l, @p padding_r, and @p padding_kind.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *
 * Outputs:
 *  - diff_weights (#mkldnn_query_diff_weights_md, 0)
 *  - diff_bias (#mkldnn_query_diff_weights_md, 1), if created with bias
 */
mkldnn_status_t MKLDNN_API mkldnn_dilated_deconvolution_backward_weights_desc_init(
        mkldnn_deconvolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *diff_weights_desc,
        const mkldnn_memory_desc_t *diff_bias_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t dilates, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** @} */

/** @addtogroup c_api_shuffle Shuffle
 * A primitive to shuffle data along the axis.
 * @{ */

/** Initializes a @p shuffle_desc for forward propagation using @p prop_kind,
 * memory descriptor @p data_desc, @p axis, and @p group_size.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 *
 */
mkldnn_status_t MKLDNN_API mkldnn_shuffle_forward_desc_init(
        mkldnn_shuffle_desc_t *shuffle_desc, mkldnn_prop_kind_t prop_kind,
        const mkldnn_memory_desc_t *data_desc, int axis,
        mkldnn_dim_t group_size);

/** Initializes a @p shuffle_desc for backward propagation using memory
 * descriptor @p diff_data_desc, @p axis, and @p group_size.
 *
 *
 * Inputs:
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 *
 */
mkldnn_status_t MKLDNN_API mkldnn_shuffle_backward_desc_init(
        mkldnn_shuffle_desc_t *shuffle_desc,
        const mkldnn_memory_desc_t *diff_data_desc, int axis,
        mkldnn_dim_t group_size);

/** @} */

/** @addtogroup c_api_eltwise Eltwise
 * A primitive to compute element-wise operations like parametric rectifier
 * linear unit (ReLU).
 *
 * Both forward and backward passes support in-place operation; that is, src
 * and dst point to the same memory for forward pass, and diff_dst and diff_src
 * point to the same memory for backward pass.
 *
 * @warning Because the original src is required for backward pass, in-place
 * forward pass in general cannot be applied during training. However, for some
 * kinds of element-wise operations (namely ReLU with alpha parameter equals 0),
 * dst and src can be interchangeable for the backward pass, which enables
 * performing in-place forward even for training.
 *
 * @{ */

/** Initializes an @p eltwise_desc for forward propagation using @p prop_kind
 * (possible values are #mkldnn_forward_training and #mkldnn_forward_inference),
 * @p alg_kind algorithm, memory descriptor @p data_desc, @p alpha, and
 * @p beta parameters.
 *
 * @sa mkldnn_eltwise_desc_t for details.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_eltwise_forward_desc_init(
        mkldnn_eltwise_desc_t *eltwise_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *data_desc,
        float alpha, float beta);

/** Initializes an @p eltwise_desc for backward propagation using @p alg_kind
 * algorithm memory descriptors @p diff_data_desc and @p data_desc, and the
 * @p alpha and @p beta parameters.
 *
 * @sa mkldnn_eltwise_desc_t for details.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_eltwise_backward_desc_init(
        mkldnn_eltwise_desc_t *eltwise_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_data_desc,
        const mkldnn_memory_desc_t *data_desc, float alpha, float beta);

/** @} */

/** @addtogroup c_api_softmax Softmax
 * A primitive to perform softmax.
 *
 * \f[dst[u][c][in] =
 *    \frac{\exp(src[ou][c][in]) - \max\limits_{c}(src[ou][c][in])}
 *    {\sum\limits_{c}\{\exp(src[ou][c][in])
 *    - \max\limits_{c}(src[ou][c][in])\}},\f]
 *
 * where \f$ou, iu\f$ are outer and inner sizes repectively, defined
 * by @p data_desc.dims and @p softmax_axis.
 * @{ */

/** Initializes a @p softmax_desc for forward propagation using @p prop_kind
 * (possible values are #mkldnn_forward_training and #mkldnn_forward_inference)
 * and memory descriptor @p data_desc.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_softmax_forward_desc_init(
        mkldnn_softmax_desc_t *softmax_desc, mkldnn_prop_kind_t prop_kind,
        const mkldnn_memory_desc_t *data_desc, int softmax_axis);

/** Initializes a @p softmax_desc for backward propagation using memory
 * descriptors @p diff_desc and @p data_desc.
 *
 * Inputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_softmax_backward_desc_init(
        mkldnn_softmax_desc_t *softmax_desc,
        const mkldnn_memory_desc_t *diff_desc,
        const mkldnn_memory_desc_t *data_desc, int softmax_axis);

/** @} */

/** @addtogroup c_api_pooling Pooling
 * A primitive to perform max or average pooling.
 *
 * Max pooling:
 * \f[dst[n][oc][oh][ow] =
 *     \max\limits_{kw,kh}
 *     (src[n][ic][oh \cdot s_h - p_l[0] + kh][ow \cdot s_w - p_r[1] + kw]),\f]
 *
 * Average pooling:
 * \f[dst[n][oc][oh][ow] =
 *     \frac{1}{KW \cdot KH}\sum\limits_{kw,kh}
 *     src[n][ic][oh \cdot s_h - p_l[0] + kh][ow \cdot s_w - p_r[1] + kw],\f]
 *
 * where \f$p_l, p_r\f$ are @p padding_l and @p padding_r respectively, and
 * output spatial dimensions are calculated similarly to how they are done in
 * convolution.
 *
 * During training, max pooling requires a workspace on forward
 * (#mkldnn_forward_training) and backward (#mkldnn_backward) passes to
 * save indices where maximum was found. The workspace layout is opaque, and
 * the indices cannot be restored from it. However, one can use backward
 * pooling to perform up-sampling (used in some detection topologies).
 *
 * @{ */

/** Initializes a pooling descriptor @p pool_desc for forward propagation using
 * @p prop_kind (possible values are #mkldnn_forward_training and
 * #mkldnn_forward_inference), @p alg_kind, memory descriptors, and pooling
 * parameters in the spatial domain: @p strides, @p kernel sizes, @p padding_l,
 * @p padding_r, and @p padding_kind.
 *
 * @note If @p padding_r is @c NULL, the padding is supposed to be symmetric.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 *  - workspace (#mkldnn_query_workspace_md, 0),
 *      if @p alg_kind = #mkldnn_pooling_max and
 *      @p prop_kind = #mkldnn_forward_training
 */
mkldnn_status_t MKLDNN_API mkldnn_pooling_forward_desc_init(
        mkldnn_pooling_desc_t *pool_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t kernel, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** Initializes a pooling descriptor @p pool_desc for backward propagation
 * using @p alg_kind, memory descriptors, and pooling parameters in the spatial
 * domain: @p strides, @p kernel sizes, @p padding_l, @p padding_r, and @p
 * padding_kind.
 *
 * @note If @p padding_r is @c NULL, the padding is supposed to be symmetric.
 *
 * Inputs:
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *  - workspace (#mkldnn_query_workspace_md, 0),
 *      if @p alg_kind = #mkldnn_pooling_max
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_pooling_backward_desc_init(
        mkldnn_pooling_desc_t *pool_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t kernel, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** @} */

/** @addtogroup c_api_lrn LRN
 * A primitive to perform local response normalization (LRN) across or within
 * channels.
 *
 * LRN accross channels:
 * \f[dst[n][c][h][w] = \left\{k + \frac{\alpha}{n_{l}}
 *                      \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2}
 *                      (src[n][c+i][h][w])^2\right\}^{-\beta}
 *                      src[n][c][h][w],\f]
 *
 * LRN within channels:
 * \f[dst[n][c][h][w] = \left\{k + \frac{\alpha}{n_{l}}
 *                      \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2}
 *                      (src[n][c][h+i][w+i])^2\right\}^{-\beta}
 *                      src[n][c][h][w],\f]
 *
 * where \f$n_{l}\f$ is the @p local_size.
 *
 * During training, LRN might or might not require a workspace on forward
 * (#mkldnn_forward_training) and backward (#mkldnn_backward) passes. The
 * behavior is implementation specific. Optimized implementations typically
 * require a workspace and use it to save some intermediate results from the
 * forward pass that accelerate computations on the backward pass.
 *
 * To check whether a workspace is required, query the LRN primitive descriptor
 * for the workspace (#mkldnn_query_workspace_md). Success indicates that the
 * workspace is required and its description will be returned.
 * @sa mkldnn_primitive_desc_query and mkldnn_primitive_desc_query_pd
 *
 * @{ */

/** Initializes an @p lrn_desc for forward propagation using @p prop_kind
 * (possible values are #mkldnn_forward_training and #mkldnn_forward_inference),
 * @p alg_kind, memory descriptor @p data_desc, and regularization
 * parameters @p local_size, @p alpha, @p beta, and @p k.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 *  - workspace (#mkldnn_query_workspace_md, 0),
 *      if the underlying implementation requires
 */
mkldnn_status_t MKLDNN_API mkldnn_lrn_forward_desc_init(
        mkldnn_lrn_desc_t *lrn_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *data_desc,
        mkldnn_dim_t local_size, float alpha, float beta, float k);

/** Initializes an @p lrn_desc for backward propagation using @p alg_kind,
 * memory descriptors @p data_desc and @p diff_data_desc, and regularization
 * parameters @p local_size, @p alpha, @p beta, and @p k.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *  - workspace (#mkldnn_query_workspace_md, 0),
 *      if the underlying implementation requires
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_lrn_backward_desc_init(
        mkldnn_lrn_desc_t *lrn_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_data_desc,
        const mkldnn_memory_desc_t *data_desc, mkldnn_dim_t local_size,
        float alpha, float beta, float k);

/** @} */

/** @addtogroup c_api_batch_normalization Batch Normalization
 * A primitive to perform batch normalization.
 *
 * \f[dst[n][c][h][w] = \gamma[c] \frac{src[n][c][h][w] - \mu[c]}
 *                      {\sqrt{\sigma[c] + eps}} + \beta[c],\f]
 *
 * where \f$\gamma[c], \beta[c]\f$ are weights and bias for a channel and,
 *
 * \f$\mu[c] = \frac{1}{NHW} \sum\limits_{whn} src[n][c][h][w]\f$,
 * \f$\sigma[c] = \frac{1}{NHW} \sum\limits_{whn}
 *                              (src[n][c][h][w] - \mu[c])^2\f$,
 *
 * and @c eps is a constant to improve numerical stability.
 *
 * Both forward and backward passes support in-place operation; that is, src
 * and dst point to the same memory for forward pass, and diff_dst and diff_src
 * point to the same memory for backward pass.
 *
 * Batch normalization supports different flavors controlled by
 * mkldnn_batch_normalization_desc_t. For example, batch normalization can
 * compute the mean and variance on its own or take them as inputs. It can
 * either perform scaling and shifting using gamma and beta parameters or not.
 * Optionally it can also perform a fused ReLU, which in case of training would
 * also require a workspace.
 *
 * @sa mkldnn_batch_normalization_desc_t
 * @{ */

/** Initializes a batch normalization descriptor @p bnrm_desc for forward
 * propagation using @p prop_kind (possible values are
 * #mkldnn_forward_training and #mkldnn_forward_inference), memory descriptor
 * @p data_desc, normalization parameter @p epsilon, and @p flags set using bit
 * flags of type mkldnn_batch_normalization_desc_t.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - mean (#mkldnn_query_src_md, 1),
 *      if #mkldnn_use_global_stats bit-flags is set in @p flags
 *  - variance (#mkldnn_query_src_md, 2),
 *      if #mkldnn_use_global_stats bit-flags is set in @p flags
 *  - scale_and_shift (#mkldnn_query_weights_md, 0),
 *      if #mkldnn_use_scaleshift bit-flags is set in @p flags
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 *  - mean (#mkldnn_query_dst_md, 1),
 *      if #mkldnn_use_global_stats bit-flags is not set in @p flags
 *      @p prop_kind = #mkldnn_forward_training
 *  - variance (#mkldnn_query_dst_md, 2),
 *      if #mkldnn_use_global_stats bit-flags is not set in @p flags
 *      and @p prop_kind = #mkldnn_forward_training
 *  - workspace (#mkldnn_query_workspace_md, 0),
 *      if #mkldnn_fuse_bn_relu bit-flags is set in @p flags
 *      and @p prop_kind = #mkldnn_forward_training
 *
 * @note In-place operation is supported; that is, dst points to the same memory
 *       as src.
 *
 * @sa mkldnn_batch_normalization_desc_t
 */
mkldnn_status_t MKLDNN_API mkldnn_batch_normalization_forward_desc_init(
        mkldnn_batch_normalization_desc_t *bnrm_desc,
        mkldnn_prop_kind_t prop_kind, const mkldnn_memory_desc_t *data_desc,
        float epsilon, unsigned flags);

/** Initializes a batch normalization descriptor @p bnrm_desc for backward
 * propagation with respect to data and scale-shift parameters using memory
 * descriptors @p data_desc and @p diff_data_desc, normalization parameter
 * @p epsilon, and @p flags set using bit flags of type
 * mkldnn_batch_normalization_desc_t.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - mean (#mkldnn_query_src_md, 1)
 *  - variance (#mkldnn_query_src_md, 2)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *  - scale_and_shift (#mkldnn_query_weights_md, 0),
 *      if #mkldnn_use_scaleshift bit-flags is set in @p flags
 *  - workspace (#mkldnn_query_workspace_md, 0),
 *      if #mkldnn_fuse_bn_relu bit-flags is set in @p flags
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 *  - diff_scale_and_shift (#mkldnn_query_diff_weights_md, 0),
 *      if #mkldnn_use_scaleshift bit-flags is set in @p flags
 *      and @p prop_kind = #mkldnn_backward
 *
 * @note in-place operation is supported,
 *       i.e. diff_src points to the same memory as diff_dst.
 *
 * @sa mkldnn_batch_normalization_desc_t
 */
mkldnn_status_t MKLDNN_API mkldnn_batch_normalization_backward_desc_init(
        mkldnn_batch_normalization_desc_t *bnrm_desc,
        mkldnn_prop_kind_t prop_kind,
        const mkldnn_memory_desc_t *diff_data_desc,
        const mkldnn_memory_desc_t *data_desc,
        float epsilon, unsigned flags);

/** @} */

/** @addtogroup c_api_inner_product Inner product
 * A primitive to compute an inner product.
 *
 * Inner product layer is also known as fully connected layer.
 * With spatial dimension:
 *
 * \f[dst[n][oc] = \sum\limits_{ic, kh, kw}
 *                 src[n][ic][kh][kw] \cdot weights[oc][ic][kh][kw]
 *                 + bias[oc]\f]
 * @{ */

/** Initializes an inner product descriptor @p ip_desc for forward propagation
 * using @p prop_kind (possible values are #mkldnn_forward_training and
 * #mkldnn_forward_inference) and memory descriptors. In order to create an
 * inner product without bias, @p bias_desc should be either @c NULL or a
 * pointer to a descriptor with memory format kind equals
 * #mkldnn_format_kind_undef.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *  - bias (#mkldnn_query_weights_md, 1), if created with bias
 *
 * Outputs:
 *  - dst (#mkldnn_query_dst_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_forward_desc_init(
        mkldnn_inner_product_desc_t *ip_desc, mkldnn_prop_kind_t prop_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc);

/** Initializes an inner product descriptor @p ip_desc for backward propagation
 * with respect to data using memory descriptors.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *  - weights (#mkldnn_query_weights_md, 0)
 *
 * Outputs:
 *  - diff_src (#mkldnn_query_diff_src_md, 0)
 */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_backward_data_desc_init(
        mkldnn_inner_product_desc_t *ip_desc,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *diff_dst_desc);

/** Initializes an inner product descriptor @p ip_desc for backward propagation
 * with respect to weights using memory descriptors.
 *
 * @note Memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src (#mkldnn_query_src_md, 0)
 *  - diff_dst (#mkldnn_query_diff_dst_md, 0)
 *
 * Outputs:
 *  - diff_weights (#mkldnn_query_diff_weights_md, 0)
 *  - diff_bias (#mkldnn_query_diff_weights_md, 1), if created with bias
 */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_backward_weights_desc_init(
        mkldnn_inner_product_desc_t *ip_desc,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *diff_weights_desc,
        const mkldnn_memory_desc_t *diff_bias_desc,
        const mkldnn_memory_desc_t *diff_dst_desc);

/** @} */

/** @addtogroup c_api_rnn RNN
 * A primitive to compute the common recurrent layer.
 * @todo add additional description for the group
 * @{ */

/**
 * Initializes a recurrent cell descriptor @p rnn_cell_desc
 * using @p rnn_cell_desc, @p kind (possible values are
 *  #mkldnn_vanilla_rnn, #mkldnn_vanilla_lstm, #mkldnn_vanilla_gru, and
 *  #mkldnn_gru_linear_before_reset),
 *  @p f (possible values are #mkldnn_eltwise_relu and
 *   #mkldnn_eltwise_tanh), @p flags, @p alpha, and @p clipping.
 */
mkldnn_status_t MKLDNN_API mkldnn_rnn_cell_desc_init(
        mkldnn_rnn_cell_desc_t *rnn_cell_desc,
        mkldnn_alg_kind_t kind, mkldnn_alg_kind_t f,
        unsigned int flags, float alpha, float clipping);

/** Returns the number of gates of a particular @p rnn_cell_desc. */
int MKLDNN_API mkldnn_rnn_cell_get_gates_count(
        const mkldnn_rnn_cell_desc_t *rnn_cell_desc);

/** Returns the number of states of a particular @p rnn_cell_desc. */
int MKLDNN_API mkldnn_rnn_cell_get_states_count(
        const mkldnn_rnn_cell_desc_t *rnn_cell_desc);

/** Sets quantization @p scale and @p shift for RNN data tensors.
 *  For performance reasons, low precision configuration of RNN primitive
 *  expects input activations to have unsigned int8 data type. Scale and shift
 *  used to quantize floating point data to unsigned integer must be passed to
 *  RNN primitive using attributes.
 *  Example usage:
 * @code
 *      // rnn parameters
 *      int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
 *      // activations quantization parameters
 *      float scale = ..., shift = ..;
 *
 *      mkldnn_primitive_attr_t rnn_attr;
 *      // create default attributes
 *      mkldnn_primitive_attr_create(&rnn_attr);
 *
 *      // set scale and shift for int8 quantization of activation
 *      mkldnn_primitive_attr_set_rnn_data_qparams(rnn_attr, scale, shift);
 *
 *      // create & configure rnn op_desc
 *      mkldnn_rnn_desc_t rnn_d;
 *      mkldnn_primitive_desc_t rnn_pd;
 *      mkldnn_primitive_desc_create(&rnn_pd, &rnn_d, attr, engine, NULL);
 * @endcode
 * @note
 *      Quantization scale and shift are common for src_layer, src_iter,
 *      dst_iter and dst_layer.
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_set_rnn_data_qparams(
        mkldnn_primitive_attr_t attr, const float scale, const float shift);

/** Sets quantization scales @p weights_scales for RNN weights tensors.
 * Low precision configuration of RNN primitive expects input weights to have
 * signed int8 data type. Scales used to quantize floating point data
 * to signed integer must be passed to RNN primitive using attributes.
 * The @p mask argument defines correspondence between output tensor dimensions
 * and the @p weights_scales array. Set i-th bit of @p mask to 1 to use
 * dedicated scaling factor for each slice of the output tensor over i-th
 * dimension. Set @p mask to 0 to use common scaling factor for the whole output
 * tensor. Example usage:
 * @code
 *      // rnn parameters
 *      int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
 *      // unique output scales per output channel
 *      float weights_scales[dic * n_gates] = { ... };
 *      // mask that specifies last two dimensions of ldigo format
 *      int mask = 0x3;
 *
 *      mkldnn_primitive_attr_t attr;
 *      // create default attributes
 *      mkldnn_primitive_attr_create(&attr);
 *
 *      // set output channel-wise weights scales
 *      mkldnn_primitive_attr_set_rnn_weights_qparams(attr, dic * n_gates, mask,
 *              weights_scales);
 *
 *      // create & configure rnn op_desc
 *      mkldnn_rnn_desc_t rnn_d;
 *      mkldnn_primitive_desc_t rnn_pd;
 *      mkldnn_primitive_desc_create(&rnn_pd, &rnn_d, attr, engine, NULL);
 * @endcode
 * @note
 *      The dimension order is always native and does not depend on the actual
 *      layout used. For example, 5 dimensional weights always have
 *      (l, d, i, g, o) logical dimension ordering.
 * @note
 *      Quantization sales are common for weights_layer and weights_iteration
 * @note
 *      There is no way to check that @p count corresponds to @p mask until an
 *      actual primitive descriptor is created, so it is user's responsibility
 *      to set proper values. The following formula must be held:
 *
 *      \f[count = \prod\limits_{d \in mask} output.dims[d]\f]
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_attr_set_rnn_weights_qparams (
        mkldnn_primitive_attr_t attr, mkldnn_dim_t count, int mask,
                const float *weights_scales);

/** Initializes a rnn descriptor @p rnn_desc for forward propagation
 * using @p prop_kind, @p rnn_cell_desc, @p direction, and memory descriptors.
 * @note If @p prop_kind equals #mkldnn_forward_training, you must query a
 * workspace memory descriptor before creating the primitive.
 *
 * @p src_iter_desc, @p bias_desc, and @p dst_iter_desc are allowed to either be
 * @c NULL or point to a zero memory descriptor, which would indicate that the
 * RNN primitive should not use them.
 *
 * @note All memory descriptors except @p src_iter_desc are allowed to be
 *       initialized with #mkldnn_format_kind_any value of @p format_kind.
 *
 * Inputs:
 *  - src_layer (#mkldnn_query_src_md, 0)
 *  - src_iter (#mkldnn_query_src_md, 1), if used
 *  - weights_layer (#mkldnn_query_weights_md, 0)
 *  - weights_iter (#mkldnn_query_weights_md, 1)
 *  - bias (#mkldnn_query_weights_md, 2), if used
 *
 * Outputs:
 *  - dst_layer (#mkldnn_query_dst_md, 0)
 *  - dst_iter (#mkldnn_query_dst_md, 1), if used
 *  - workspace (#mkldnn_query_workspace_md, 0),
 *      if @p prop_kind equals #mkldnn_forward_training
 */
mkldnn_status_t MKLDNN_API mkldnn_rnn_forward_desc_init(
        mkldnn_rnn_desc_t *rnn_desc, mkldnn_prop_kind_t prop_kind,
        const mkldnn_rnn_cell_desc_t *rnn_cell_desc,
        const mkldnn_rnn_direction_t direction,
        const mkldnn_memory_desc_t *src_layer_desc,
        const mkldnn_memory_desc_t *src_iter_desc,
        const mkldnn_memory_desc_t *weights_layer_desc,
        const mkldnn_memory_desc_t *weights_iter_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_layer_desc,
        const mkldnn_memory_desc_t *dst_iter_desc);

/** Initializes a rnn descriptor @p rnn_desc for backward propagation
 * using @p prop_kind, @p rnn_cell_desc, @p direction, and memory descriptors.
 *
 * @note All memory descriptors are allowed to be initialized with
 *       #mkldnn_format_kind_any value of @p format_kind.
 *
 * @p src_iter_desc (simultaneously with @p diff_src_iter_desc),
 * @p bias_desc (simultaneously with @p diff_bias_desc), and
 * @p dst_iter_desc (simultaneously with @p diff_src_iter_desc) are allowed to
 * either be @c NULL or point to a zero memory descriptor, which would indicate
 * that the RNN primitive should not use them.
 *
 * Inputs:
 *  - src_layer (#mkldnn_query_src_md, 0)
 *  - src_iter (#mkldnn_query_src_md, 1), if used
 *  - weights_layer (#mkldnn_query_weights_md, 0)
 *  - weights_iter (#mkldnn_query_weights_md, 1)
 *  - bias (#mkldnn_query_weights_md, 2), if used
 *  - dst_layer (#mkldnn_query_dst_md, 0)
 *  - dst_iter (#mkldnn_query_dst_md, 1), if used
 *  - diff_dst_layer (#mkldnn_query_diff_dst_md, 0)
 *  - diff_dst_iter (#mkldnn_query_diff_dst_md, 1), if used
 *  - workspace (#mkldnn_query_workspace_md, 0)
 *
 * Outputs:
 *  - diff_src_layer (#mkldnn_query_diff_src_md, 0)
 *  - diff_src_iter (#mkldnn_query_diff_src_md, 1), if used
 *  - diff_weights_layer (#mkldnn_query_diff_weights_md, 0)
 *  - diff_weights_iter (#mkldnn_query_diff_weights_md, 1)
 *  - diff_bias (#mkldnn_query_diff_weights_md, 2), if used
 */
mkldnn_status_t MKLDNN_API mkldnn_rnn_backward_desc_init(
        mkldnn_rnn_desc_t *rnn_desc, mkldnn_prop_kind_t prop_kind,
        const mkldnn_rnn_cell_desc_t *rnn_cell_desc,
        const mkldnn_rnn_direction_t direction,
        const mkldnn_memory_desc_t *src_layer_desc,
        const mkldnn_memory_desc_t *src_iter_desc,
        const mkldnn_memory_desc_t *weights_layer_desc,
        const mkldnn_memory_desc_t *weights_iter_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_layer_desc,
        const mkldnn_memory_desc_t *dst_iter_desc,
        const mkldnn_memory_desc_t *diff_src_layer_desc,
        const mkldnn_memory_desc_t *diff_src_iter_desc,
        const mkldnn_memory_desc_t *diff_weights_layer_desc,
        const mkldnn_memory_desc_t *diff_weights_iter_desc,
        const mkldnn_memory_desc_t *diff_bias_desc,
        const mkldnn_memory_desc_t *diff_dst_layer,
        const mkldnn_memory_desc_t *diff_dst_iter_desc);

/** @} */

/** @} */

/** @addtogroup c_api_engine Engine operations
 * @{ */

/** Returns the number of engines of a particular @p kind. */
size_t MKLDNN_API mkldnn_engine_get_count(mkldnn_engine_kind_t kind);

/** Creates an @p engine of particular @p kind and @p index. */
mkldnn_status_t MKLDNN_API mkldnn_engine_create(mkldnn_engine_t *engine,
        mkldnn_engine_kind_t kind, size_t index);

/** Returns the kind of an @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_engine_get_kind(mkldnn_engine_t engine,
        mkldnn_engine_kind_t *kind);

/** Destroys an @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_engine_destroy(mkldnn_engine_t engine);

/** @} */

/** @addtogroup c_api_stream Execution stream operations
 * @{ */

/** Creates an execution @p stream for @p engine and with @p flags. */
mkldnn_status_t MKLDNN_API mkldnn_stream_create(mkldnn_stream_t *stream,
        mkldnn_engine_t engine, unsigned flags);

/** Destroys an execution @p stream. */
mkldnn_status_t MKLDNN_API mkldnn_stream_destroy(mkldnn_stream_t stream);

/** @} */

/** @addtogroup c_api_service Service functions
 * @{ */

/** Sets verbosity level (print information to stdout).
 * Possible levels are:
 *  - 0 -- no verbose output (default)
 *  - 1 -- primitive information at execution
 *  - 2 -- primitive information at creation and execution
 *
 * @note
 *     Dumping information might affect performance.
 *     This setting overrides the MKLDNN_VERBOSE environment variable. */
mkldnn_status_t MKLDNN_API mkldnn_set_verbose(int level);

/** Enables or disables dumping of JIT-generated code.
 * The enable parameter can be:
 *  - 0 -- disable
 *  - any other value -- enable
 *
 * @note
 *     This setting overrides the MKLDNN_JIT_DUMP environment variable. */
mkldnn_status_t MKLDNN_API mkldnn_set_jit_dump(int enable);

/** Gets library version information.
 * Version information includes:
 *  - major -- major version number
 *  - minor -- minor version number
 *  - patch -- patch release number
 *  - hash -- git commit hash */
const mkldnn_version_t MKLDNN_API *mkldnn_version();

/** @} */

/** @addtogroup c_api_blas BLAS functions
 * A subset of Basic Linear ALgebra (BLAS) functions to perform
 * matrix-matrix multiplication.
 * @{ */

/** SGEMM performs a matrix-matrix multiplication operation defined as
 *
 * C := alpha*op( A )*op( B ) + beta*C
 *
 * where
 *  - op( X ) is one of op( X ) = X or op( X ) = X**T,
 *  - alpha and beta are scalars,
 *  - A, B and C are matrices, with op( A ) an m by k matrix, op( B ) a k by n matrix
 *    and C an m by n matrix.
 *
 * The matrices are assumed to be stored in column-major order (the elements
 * in a matrix columns are contiguous in memory).
 *
 * @note
 *      The API is different from the standard BLAS routine
 *      because it returns mkldnn_status_t for error handling.
 *      XERBLA is not supported: no error message will be printed
 *      in case of incorrect parameters. */
mkldnn_status_t MKLDNN_API mkldnn_sgemm(
        const char *transa, const char *transb,
        const mkldnn_dim_t *M, const mkldnn_dim_t *N, const mkldnn_dim_t *K,
        const float *alpha, const float *A, const mkldnn_dim_t *lda,
        const float *B, const mkldnn_dim_t *ldb,
        const float *beta, float *C, const mkldnn_dim_t *ldc);

/** gemm_s8u8s32 and gemm_s8s8s32 perform a matrix-matrix multiplication
 * operation and add the result to a scalar-matrix product. For the final
 * result, a vector is added to each row or column of the output matrix.
 * The operation is defined as:
 *
 * C := alpha*(op(A) + A_offset) * (op(B) + B_offset) + beta*C + C_offset
 *
 * where
 *  - op( X ) = X or op( X ) = X**T,
 *  - A_offset is an m-by-k matrix with every element equal to the value oa,
 *  - B_offset is an k-by-n matrix with every element equal to the value ob,
 *  - C_offset is an m-by-n matrix defined by the oc array, size len:
 *    - if offsetc = F: len must be at least 1
 *    - if offsetc = C: len must be at least max(1, m)
 *    - if offsetc = R: len must be at least max(1, n)
 *  - alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *    an m-by-k matrix, op( B ) a k-by-n matrix and C an m-by-n matrix.
 *
 * The matrices are assumed to be stored in column-major order (the elements
 * in a matrix columns are contiguous in memory).
 *
 * @note
 *      The API is different compared with the standard BLAS routine
 *      because it returns mkldnn_status_t for error handling.
 *      XERBLA is not supported: no error message will be printed
 *      in case of incorrect parameters. */
mkldnn_status_t MKLDNN_API mkldnn_gemm_s8u8s32(
        const char *transa, const char *transb, const char *offsetc,
        const mkldnn_dim_t *M, const mkldnn_dim_t *N, const mkldnn_dim_t *K,
        const float *alpha,
        const int8_t *A, const mkldnn_dim_t *lda, const int8_t *ao,
        const uint8_t *B, const mkldnn_dim_t *ldb, const int8_t *bo,
        const float *beta,
        int32_t *c, const mkldnn_dim_t *ldc, const int32_t *co);

mkldnn_status_t MKLDNN_API mkldnn_gemm_s8s8s32(
        const char *transa, const char *transb, const char *offsetc,
        const mkldnn_dim_t *M, const mkldnn_dim_t *N, const mkldnn_dim_t *K,
        const float *alpha,
        const int8_t *A, const mkldnn_dim_t *lda, const int8_t *ao,
        const int8_t *B, const mkldnn_dim_t *ldb, const int8_t *bo,
        const float *beta,
        int32_t *c, const mkldnn_dim_t *ldc, const int32_t *co);
/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif
