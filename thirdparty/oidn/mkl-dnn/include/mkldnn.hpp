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

#ifndef MKLDNN_HPP
#define MKLDNN_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stdlib.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iterator>

#include "mkldnn.h"
#endif

namespace mkldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_api_utils Utils
/// @{

/// A class that provides the destructor for an Intel(R) MKL-DNN C handle
template <typename T> class handle_traits {};

/// A class for wrapping an Intel(R) MKL-DNN handle. It is used as the base
/// class for primitive (#mkldnn_primitive_t), engine (#mkldnn_engine_t), and
/// stream (#mkldnn_stream_t) handles. An object of the #mkldnn::handle class
/// can be passed by value. This class enables wrapping:
///  - Newly constructed handles.
///    @n In this case, the constructed handle uses reference counting provided
///    by @p std::shared_ptr with a proper deleter function specified through
///    the @p handle_traits class.
///  - Pre-existing handles returned by the Intel(R) MKL-DNN C API (for
///    example, through mkldnn_primitive_get_primitive_desc()).
///    @n In this case, an Intel(R) MKL-DNN C API handle is wrapped without a
///    deleter because it is assumed that the handle wrapper for the original
///    object deletes the handle (this model is similar to @p std::weak_ptr).
template <typename T, typename traits=handle_traits<T>> class handle {
private:
    std::shared_ptr<typename std::remove_pointer<T>::type> _data;
    handle(const handle &&) = delete;
    handle &operator=(const handle &&other) = delete;
protected:
    bool operator==(const T other) const { return other == _data.get(); }
    bool operator!=(const T other) const { return !(*this == other); }
public:
    /// Constructs a C handle wrapper.
    /// @param t The C handle to wrap.
    /// @param weak A flag to specify whether to construct a weak wrapper.
    handle(T t = 0, bool weak = false): _data(0) {
        reset(t, weak);
    }

    handle(const handle &other): _data(other._data) {}
    handle &operator=(const handle &other) {
        _data = other._data;
        return *this;
    }
    /// Resets the value of a C handle.
    /// @param t The new value of the C handle.
    /// @param weak A flag to specify whether the wrapper should be weak.
    void reset(T t, bool weak = false) {
        auto dummy_destructor = [](T) { return decltype(traits::destructor(0))(0); };
        _data.reset(t, weak ? dummy_destructor : traits::destructor);
    }

    /// Returns the value of the underlying C handle.
    T get() const { return _data.get(); }

    bool operator==(const handle &other) const { return other._data.get() == _data.get(); }
    bool operator!=(const handle &other) const { return !(*this == other); }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_memory_t> {
    static constexpr auto destructor = &mkldnn_memory_destroy;
};

template <> struct handle_traits<mkldnn_primitive_desc_t> {
    static constexpr auto destructor = &mkldnn_primitive_desc_destroy;
};

template <> struct handle_traits<mkldnn_primitive_t> {
    static constexpr auto destructor = &mkldnn_primitive_destroy;
};

template <> struct handle_traits<mkldnn_primitive_desc_iterator_t> {
    static constexpr auto destructor = &mkldnn_primitive_desc_iterator_destroy;
};
#endif

struct memory;
struct primitive_desc;

/// Base class for all computational primitives.
class primitive: public handle<mkldnn_primitive_t> {
    friend struct error;
    friend struct stream;
    using handle::handle;
public:
    /// A proxy to C primitive kind enum
    enum class kind {
        undefined_primitive = mkldnn_undefined_primitive,
        reorder = mkldnn_reorder,
        concat = mkldnn_concat,
        sum = mkldnn_sum,
        convolution = mkldnn_convolution,
        deconvolution = mkldnn_deconvolution,
        shuffle = mkldnn_shuffle,
        eltwise = mkldnn_eltwise,
        softmax = mkldnn_softmax,
        pooling = mkldnn_pooling,
        lrn = mkldnn_lrn,
        batch_normalization = mkldnn_batch_normalization,
        inner_product = mkldnn_inner_product,
        rnn = mkldnn_rnn,
    };

    primitive(const_mkldnn_primitive_desc_t c_pd);
    primitive(const primitive_desc &pd);

    /// Returns the descriptor of the underlying C API primitive.
    inline const_mkldnn_primitive_desc_t get_primitive_desc() const;
    // TODO: use the C++ API wrapper structure.

    void execute(struct stream &astream,
            const std::unordered_map<int, memory> &args) const;
};

inline mkldnn_primitive_kind_t convert_to_c(primitive::kind akind) {
    return static_cast<mkldnn_primitive_kind_t>(akind);
}
/// Intel(R) MKL-DNN exception class.
///
/// This class captures the status returned by the failed C API function, error
/// message, and, optionally, handle of the primitive that caused the error.
struct error: public std::exception {
    mkldnn_status_t status;
    const char *message;

    /// Constructs an error instance.
    ///
    /// @param astatus The error status returned by the C API.
    /// @param amessage The error message.
    error(mkldnn_status_t astatus, const char *amessage)
        : status(astatus), message(amessage) {}

    /// A convenience function for wrapping calls to the C API. Checks the
    /// return status and throws an #error in case of failure.
    ///
    /// @param status The error status returned by the C API.
    /// @param message The error message.
    static void wrap_c_api(mkldnn_status_t status, const char *message) {
        if (status != mkldnn_success)
            throw error(status, message);
    }
};

const_mkldnn_primitive_desc_t primitive::get_primitive_desc() const {
    const_mkldnn_primitive_desc_t pd;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(), &pd),
            "could not get primitive descriptor by primitive");
    return pd;
}
/// @}

/// @addtogroup cpp_api_enums Common data types and enumerations
/// A proxy to @ref c_api_types in @ref c_api.
///
/// @{

enum scratchpad_mode {
    scratchpad_mode_library = mkldnn_scratchpad_mode_library,
    scratchpad_mode_user = mkldnn_scratchpad_mode_user,
};

inline mkldnn_scratchpad_mode_t convert_to_c(scratchpad_mode mode) {
    return static_cast<mkldnn_scratchpad_mode_t>(mode);
}

enum padding_kind {
    zero = mkldnn_padding_zero
};

inline mkldnn_padding_kind_t convert_to_c(padding_kind kind) {
    return static_cast<mkldnn_padding_kind_t>(kind);
}

enum prop_kind {
    forward_training = mkldnn_forward_training,
    forward_scoring = mkldnn_forward_scoring,
    forward_inference = mkldnn_forward_inference,
    forward = mkldnn_forward,
    backward = mkldnn_backward,
    backward_data = mkldnn_backward_data,
    backward_weights = mkldnn_backward_weights,
    backward_bias = mkldnn_backward_bias
};

inline mkldnn_prop_kind_t convert_to_c(prop_kind kind) {
    return static_cast<mkldnn_prop_kind_t>(kind);
}

enum algorithm {
    algorithm_undef = mkldnn_alg_kind_undef,
    convolution_auto = mkldnn_convolution_auto,
    convolution_direct = mkldnn_convolution_direct,
    convolution_winograd = mkldnn_convolution_winograd,
    deconvolution_direct = mkldnn_deconvolution_direct,
    deconvolution_winograd = mkldnn_deconvolution_winograd,
    eltwise_relu = mkldnn_eltwise_relu,
    eltwise_tanh = mkldnn_eltwise_tanh,
    eltwise_elu = mkldnn_eltwise_elu,
    eltwise_square = mkldnn_eltwise_square,
    eltwise_abs = mkldnn_eltwise_abs,
    eltwise_sqrt = mkldnn_eltwise_sqrt,
    eltwise_linear = mkldnn_eltwise_linear,
    eltwise_bounded_relu = mkldnn_eltwise_bounded_relu,
    eltwise_soft_relu = mkldnn_eltwise_soft_relu,
    eltwise_logistic = mkldnn_eltwise_logistic,
    lrn_across_channels = mkldnn_lrn_across_channels,
    lrn_within_channel  = mkldnn_lrn_within_channel,
    pooling_max = mkldnn_pooling_max,
    pooling_avg = mkldnn_pooling_avg,
    pooling_avg_include_padding = mkldnn_pooling_avg_include_padding,
    pooling_avg_exclude_padding = mkldnn_pooling_avg_exclude_padding,
    vanilla_rnn = mkldnn_vanilla_rnn,
    vanilla_lstm = mkldnn_vanilla_lstm,
    vanilla_gru = mkldnn_vanilla_gru,
    gru_linear_before_reset = mkldnn_gru_linear_before_reset
};

inline mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
    return static_cast<mkldnn_alg_kind_t>(aalgorithm);
}

enum batch_normalization_flag {
    use_global_stats = mkldnn_use_global_stats,
    use_scale_shift = mkldnn_use_scaleshift,
    fuse_bn_relu = mkldnn_fuse_bn_relu
};

inline mkldnn_batch_normalization_flag_t convert_to_c(
        batch_normalization_flag aflag) {
    return static_cast<mkldnn_batch_normalization_flag_t>(aflag);
}

enum rnn_direction {
    unidirectional_left2right = mkldnn_unidirectional_left2right,
    unidirectional_right2left = mkldnn_unidirectional_right2left,
    unidirectional = mkldnn_unidirectional,
    bidirectional_concat = mkldnn_bidirectional_concat,
    bidirectional_sum = mkldnn_bidirectional_sum,
};

inline mkldnn_rnn_direction_t convert_to_c(rnn_direction adir) {
    return static_cast<mkldnn_rnn_direction_t>(adir);
}

enum query {
    undef = mkldnn_query_undef,

    query_engine = mkldnn_query_engine,
    primitive_kind = mkldnn_query_primitive_kind,

    num_of_inputs_s32 = mkldnn_query_num_of_inputs_s32,
    num_of_outputs_s32 = mkldnn_query_num_of_outputs_s32,

    time_estimate_f64 = mkldnn_query_time_estimate_f64,
    memory_consumption_s64 = mkldnn_query_memory_consumption_s64,

    query_scratchpad_engine = mkldnn_query_scratchpad_engine,

    impl_info_str = mkldnn_query_impl_info_str,

    op_d = mkldnn_query_op_d,
    convolution_d = mkldnn_query_convolution_d,
    deconvolution_d = mkldnn_query_deconvolution_d,
    shuffle_d = mkldnn_query_shuffle_d,
    eltwise_d = mkldnn_query_eltwise_d,
    softmax_d = mkldnn_query_softmax_d,
    pooling_d = mkldnn_query_pooling_d,
    lrn_d = mkldnn_query_lrn_d,
    batch_normalization_d = mkldnn_query_batch_normalization_d,
    inner_product_d = mkldnn_query_inner_product_d,
    rnn_d = mkldnn_query_rnn_d,

    src_md = mkldnn_query_src_md,
    diff_src_md = mkldnn_query_diff_src_md,
    weights_md = mkldnn_query_weights_md,
    diff_weights_md = mkldnn_query_diff_weights_md,
    dst_md = mkldnn_query_dst_md,
    diff_dst_md = mkldnn_query_diff_dst_md,
    workspace_md = mkldnn_query_workspace_md,
    scratchpad_md = mkldnn_query_scratchpad_md,
};

inline mkldnn_query_t convert_to_c(query aquery) {
    return static_cast<mkldnn_query_t>(aquery);
}

/// @}

/// @addtogroup cpp_api_attr Attributes
/// An extension for controlling primitive behavior.
///
/// @sa @ref c_api_attributes in @ref c_api
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_post_ops_t> {
    static constexpr auto destructor = &mkldnn_post_ops_destroy;
};
#endif

struct post_ops: public handle<mkldnn_post_ops_t> {
    post_ops() {
        mkldnn_post_ops_t result;
        error::wrap_c_api(mkldnn_post_ops_create(&result),
                "could not create post operation sequence");
        reset(result);
    }

    int len() const { return mkldnn_post_ops_len(get()); }

    primitive::kind kind(int index) const {
        error::wrap_c_api(
                index < len() ? mkldnn_success : mkldnn_invalid_arguments,
                "post_ops index is out of range");
        return static_cast<primitive::kind>(mkldnn_post_ops_get_kind(get(),
                    index));
    }

    void append_sum(float scale = 1.) {
        error::wrap_c_api(mkldnn_post_ops_append_sum(get(), scale),
                "could not append sum");
    }

    void get_params_sum(int index, float &scale) const {
        error::wrap_c_api(mkldnn_post_ops_get_params_sum(get(), index, &scale),
                "could not get sum params");
    }

    void append_eltwise(float scale, algorithm alg, float alpha,
            float beta) {
        error::wrap_c_api(mkldnn_post_ops_append_eltwise(get(), scale,
                    convert_to_c(alg), alpha, beta),
                "could not append eltwise");
    }

    void get_params_eltwise(int index, float &scale, algorithm &alg,
            float &alpha, float &beta) const {
        mkldnn_alg_kind_t c_alg;
        error::wrap_c_api(mkldnn_post_ops_get_params_eltwise(get(), index,
                    &scale, &c_alg, &alpha, &beta),
                "could not get eltwise params");
        alg = static_cast<algorithm>(c_alg);
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_primitive_attr_t> {
    static constexpr auto destructor = &mkldnn_primitive_attr_destroy;
};
#endif

struct primitive_attr: public handle<mkldnn_primitive_attr_t> {
    primitive_attr() {
        mkldnn_primitive_attr_t result;
        error::wrap_c_api(mkldnn_primitive_attr_create(&result),
                "could not create a primitive attr");
        reset(result);
    }

    scratchpad_mode get_scratchpad_mode() const {
        mkldnn_scratchpad_mode_t result;
        error::wrap_c_api(mkldnn_primitive_attr_get_scratchpad_mode(
                    get(), &result), "could not get scratchpad mode");
        return scratchpad_mode(result);
    }

    void set_scratchpad_mode(scratchpad_mode mode) {
        error::wrap_c_api(mkldnn_primitive_attr_set_scratchpad_mode(
                    get(), mkldnn::convert_to_c(mode)),
                "could not set scratchpad mode");
    }

    void get_output_scales(int &mask, std::vector<float> &scales) const
    {
        mkldnn_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(mkldnn_primitive_attr_get_output_scales(get(),
                    &count, &c_mask, &c_scales),
                "could not get int output scales");
        scales.resize(count);

        mask = c_mask;
        for (mkldnn_dim_t c = 0; c < count; ++c)
            scales[c] = c_scales[c];
    }

    void set_output_scales(int mask, const std::vector<float> &scales)
    {
        error::wrap_c_api(mkldnn_primitive_attr_set_output_scales(get(),
                    (mkldnn_dim_t)scales.size(), mask, &scales[0]),
                "could not set int output scales");
    }

    const post_ops get_post_ops() const {
        post_ops result;
        const_mkldnn_post_ops_t c_result;
        error::wrap_c_api(mkldnn_primitive_attr_get_post_ops(get(), &c_result),
                "could not get post operation sequence");
        result.reset(const_cast<mkldnn_post_ops_t>(c_result), true);
        return result;
    }

    void set_post_ops(post_ops ops) {
        error::wrap_c_api(mkldnn_primitive_attr_set_post_ops(get(), ops.get()),
                "could not set post operation sequence");
    }

    void set_rnn_data_qparams(const float scale, const float shift)
    {
        error::wrap_c_api(mkldnn_primitive_attr_set_rnn_data_qparams(get(),
                    scale, shift), "could not set rnn data int scale/shift");
    }

    void set_rnn_weights_qparams(int mask, const std::vector<float> &scales)
    {
        error::wrap_c_api(mkldnn_primitive_attr_set_rnn_weights_qparams(get(),
                    (int)scales.size(), mask, &scales[0]),
                "could not set rnn weights int scales");
    }
};

/// @}

/// @addtogroup cpp_api_engine Engine
/// Engine operations.
///
/// @sa @ref c_api_engine in @ref c_api
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_engine_t> {
    static constexpr auto destructor = &mkldnn_engine_destroy;
};
#endif

/// An execution engine.
struct engine: public handle<mkldnn_engine_t> {
    friend class primitive;
    // gcc bug??? using handle::handle;

    /// Kinds of engines.
    enum kind {
        /// An unspecified engine
        any = mkldnn_any_engine,
        /// CPU engine
        cpu = mkldnn_cpu,
    };

    /// Returns the number of engines of a certain kind.
    ///
    /// @param akind The kind of engines to count.

    static size_t get_count(kind akind) {
        return mkldnn_engine_get_count(convert_to_c(akind));
    }

    /// Constructs an engine.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index The index of the engine. Must be less than the value
    ///              returned by #get_count() for this particular kind of engine.

    engine(kind akind, size_t index) {
        mkldnn_engine_t aengine;
        error::wrap_c_api(
                mkldnn_engine_create(&aengine,
                    convert_to_c(akind), index),
                "could not create an engine");
        reset(aengine);
    }

    explicit engine(const mkldnn_engine_t& aengine)
        : handle(aengine, true) {}

    engine(const handle<mkldnn_primitive_desc_t> &pd) {
        mkldnn_engine_t engine_q;
        error::wrap_c_api(
                mkldnn_primitive_desc_query(pd.get(),
                    mkldnn::convert_to_c(query_engine), 0, &engine_q),
                "could not get engine from primitive_desc");
        reset(engine_q, true);
    }

    template <class primitive_desc>
    static engine query(const primitive_desc &pd) {
        mkldnn_engine_t engine_q;
        error::wrap_c_api(
                mkldnn_primitive_desc_query(pd.get(),
                    mkldnn::convert_to_c(query_engine), 0, &engine_q),
                "could not get engine from primitive_desc");

        return engine(engine_q);
    }

private:
    static mkldnn_engine_kind_t convert_to_c(kind akind) {
        return static_cast<mkldnn_engine_kind_t>(akind);
    }
};

/// @}

/// @addtogroup cpp_api_stream Stream
/// Execution stream operations
///
/// @sa @ref c_api_stream in @ref c_api
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<mkldnn_stream_t> {
    static constexpr auto destructor = &mkldnn_stream_destroy;
};
#endif

struct stream: public handle<mkldnn_stream_t> {
    using handle::handle;

    enum: unsigned {
        default_flags = mkldnn_stream_default_flags,
    };

    /// Constructs a stream.
    stream(const engine &aengine,
            unsigned flags = static_cast<unsigned>(default_flags)) {
        mkldnn_stream_t astream;
        error::wrap_c_api(mkldnn_stream_create(&astream, aengine.get(), flags),
                "could not create a stream");
        reset(astream);
    }
};

/// @}

/// @addtogroup cpp_api_memory_related Memory and memory related operations
/// @{

/// @addtogroup cpp_api_memory Memory
/// A primitive to describe and store data.
///
/// For more information, refer to @ref c_api_memory in @ref c_api.
/// @{

/// Memory that describes the data.
struct memory: public handle<mkldnn_memory_t> {
    public:
    typedef mkldnn_dim_t dim;
    typedef std::vector<dim> dims;

    template <typename T> static void validate_dims(const std::vector<T> &v) {
        if (v.size() > MKLDNN_MAX_NDIMS)
            throw error(mkldnn_invalid_arguments, "invalid dimensions");
    }

    /// Data type specification. See #mkldnn_data_type_t for a detailed
    /// description.
    enum data_type {
        data_undef = mkldnn_data_type_undef,
        f32 = mkldnn_f32,
        s32 = mkldnn_s32,
        s8 = mkldnn_s8,
        u8 = mkldnn_u8,
    };

    /// Memory format tag specification. See #mkldnn_format_tag_t
    /// for a detailed description.
    enum format_tag {
        format_tag_undef = mkldnn_format_tag_undef,
        any = mkldnn_format_tag_any,
        a = mkldnn_a,
        ab = mkldnn_ab,
        abc = mkldnn_abc,
        abcd = mkldnn_abcd,
        abcde = mkldnn_abcde,
        abcdef = mkldnn_abcdef,
        abdec = mkldnn_abdec,
        acb = mkldnn_acb,
        acbde = mkldnn_acbde,
        acdb = mkldnn_acdb,
        acdeb = mkldnn_acdeb,
        ba = mkldnn_ba,
        bac = mkldnn_bac,
        bacd = mkldnn_bacd,
        bcda = mkldnn_bcda,
        cba = mkldnn_cba,
        cdba = mkldnn_cdba,
        cdeba = mkldnn_cdeba,
        decab = mkldnn_decab,
        Abc16a = mkldnn_Abc16a,
        ABc16a16b = mkldnn_ABc16a16b,
        aBc16b = mkldnn_aBc16b,
        ABc16b16a = mkldnn_ABc16b16a,
        Abc4a = mkldnn_Abc4a,
        aBc4b = mkldnn_aBc4b,
        ABc4b16a4b = mkldnn_ABc4b16a4b,
        ABc4b4a = mkldnn_ABc4b4a,
        ABc8a16b2a = mkldnn_ABc8a16b2a,
        ABc8a8b = mkldnn_ABc8a8b,
        aBc8b = mkldnn_aBc8b,
        ABc8b16a2b = mkldnn_ABc8b16a2b,
        ABc8b8a = mkldnn_ABc8b8a,
        Abcd16a = mkldnn_Abcd16a,
        ABcd16a16b = mkldnn_ABcd16a16b,
        aBcd16b = mkldnn_aBcd16b,
        ABcd16b16a = mkldnn_ABcd16b16a,
        aBCd16b16c = mkldnn_aBCd16b16c,
        aBCd16c16b = mkldnn_aBCd16c16b,
        Abcd4a = mkldnn_Abcd4a,
        aBcd4b = mkldnn_aBcd4b,
        ABcd4b16a4b = mkldnn_ABcd4b16a4b,
        ABcd4b4a = mkldnn_ABcd4b4a,
        aBCd4c16b4c = mkldnn_aBCd4c16b4c,
        aBCd4c4b = mkldnn_aBCd4c4b,
        ABcd8a16b2a = mkldnn_ABcd8a16b2a,
        ABcd8a8b = mkldnn_ABcd8a8b,
        aBcd8b = mkldnn_aBcd8b,
        ABcd8b16a2b = mkldnn_ABcd8b16a2b,
        aBCd8b16c2b = mkldnn_aBCd8b16c2b,
        ABcd8b8a = mkldnn_ABcd8b8a,
        aBCd8b8c = mkldnn_aBCd8b8c,
        aBCd8c16b2c = mkldnn_aBCd8c16b2c,
        aBCd8c8b = mkldnn_aBCd8c8b,
        Abcde16a = mkldnn_Abcde16a,
        ABcde16a16b = mkldnn_ABcde16a16b,
        aBcde16b = mkldnn_aBcde16b,
        ABcde16b16a = mkldnn_ABcde16b16a,
        aBCde16b16c = mkldnn_aBCde16b16c,
        aBCde16c16b = mkldnn_aBCde16c16b,
        aBCde2c8b4c = mkldnn_aBCde2c8b4c,
        Abcde4a = mkldnn_Abcde4a,
        aBcde4b = mkldnn_aBcde4b,
        ABcde4b4a = mkldnn_ABcde4b4a,
        aBCde4b4c = mkldnn_aBCde4b4c,
        aBCde4c16b4c = mkldnn_aBCde4c16b4c,
        aBCde4c4b = mkldnn_aBCde4c4b,
        Abcde8a = mkldnn_Abcde8a,
        ABcde8a8b = mkldnn_ABcde8a8b,
        aBcde8b = mkldnn_aBcde8b,
        ABcde8b16a2b = mkldnn_ABcde8b16a2b,
        aBCde8b16c2b = mkldnn_aBCde8b16c2b,
        ABcde8b8a = mkldnn_ABcde8b8a,
        aBCde8b8c = mkldnn_aBCde8b8c,
        aBCde8c16b2c = mkldnn_aBCde8c16b2c,
        aBCde8c8b = mkldnn_aBCde8c8b,
        aBcdef16b = mkldnn_aBcdef16b,
        aBCdef16b16c = mkldnn_aBCdef16b16c,
        aBCdef16c16b = mkldnn_aBCdef16c16b,
        aBcdef4b = mkldnn_aBcdef4b,
        aBCdef4c4b = mkldnn_aBCdef4c4b,
        aBCdef8b8c = mkldnn_aBCdef8b8c,
        aBCdef8c16b2c = mkldnn_aBCdef8c16b2c,
        aBCdef8c8b = mkldnn_aBCdef8c8b,
        aBdc16b = mkldnn_aBdc16b,
        aBdc4b = mkldnn_aBdc4b,
        aBdc8b = mkldnn_aBdc8b,
        aBdec16b = mkldnn_aBdec16b,
        aBdec4b = mkldnn_aBdec4b,
        aBdec8b = mkldnn_aBdec8b,
        aBdefc16b = mkldnn_aBdefc16b,
        aBdefc4b = mkldnn_aBdefc4b,
        aBdefc8b = mkldnn_aBdefc8b,
        Acb16a = mkldnn_Acb16a,
        Acb4a = mkldnn_Acb4a,
        Acb8a = mkldnn_Acb8a,
        aCBd16b16c = mkldnn_aCBd16b16c,
        aCBde16b16c = mkldnn_aCBde16b16c,
        Acdb16a = mkldnn_Acdb16a,
        Acdb4a = mkldnn_Acdb4a,
        Acdb8a = mkldnn_Acdb8a,
        Acdeb16a = mkldnn_Acdeb16a,
        Acdeb4a = mkldnn_Acdeb4a,
        Acdeb8a = mkldnn_Acdeb8a,
        BAc16a16b = mkldnn_BAc16a16b,
        BAcd16a16b = mkldnn_BAcd16a16b,
        format_tag_last = mkldnn_format_tag_last,

        x = mkldnn_x,
        nc = mkldnn_nc,
        cn = mkldnn_cn,
        ncw = mkldnn_ncw,
        nwc = mkldnn_nwc,
        nchw = mkldnn_nchw,
        nhwc = mkldnn_nhwc,
        chwn = mkldnn_chwn,
        ncdhw = mkldnn_ncdhw,
        ndhwc = mkldnn_ndhwc,
        oi = mkldnn_oi,
        io = mkldnn_io,
        oiw = mkldnn_oiw,
        wio = mkldnn_wio,
        oihw = mkldnn_oihw,
        hwio = mkldnn_hwio,
        ihwo = mkldnn_ihwo,
        iohw = mkldnn_iohw,
        oidhw = mkldnn_oidhw,
        dhwio = mkldnn_dhwio,
        goiw = mkldnn_goiw,
        goihw = mkldnn_goihw,
        hwigo = mkldnn_hwigo,
        giohw = mkldnn_giohw,
        goidhw = mkldnn_goidhw,
        tnc = mkldnn_tnc,
        ntc = mkldnn_ntc,
        ldsnc = mkldnn_ldsnc,
        ldigo = mkldnn_ldigo,
        ldgoi = mkldnn_ldgoi,
        ldgo = mkldnn_ldgo,
        nCdhw16c = mkldnn_nCdhw16c,
        nCdhw4c = mkldnn_nCdhw4c,
        nCdhw8c = mkldnn_nCdhw8c,
        nChw16c = mkldnn_nChw16c,
        nChw4c = mkldnn_nChw4c,
        nChw8c = mkldnn_nChw8c,
        nCw16c = mkldnn_nCw16c,
        nCw4c = mkldnn_nCw4c,
        nCw8c = mkldnn_nCw8c,
        IOw16o16i = mkldnn_IOw16o16i,
        OIw16i16o = mkldnn_OIw16i16o,
        OIw16o16i = mkldnn_OIw16o16i,
        Oiw16o = mkldnn_Oiw16o,
        OIw4i16o4i = mkldnn_OIw4i16o4i,
        OIw4i4o = mkldnn_OIw4i4o,
        Oiw4o = mkldnn_Oiw4o,
        OIw8i16o2i = mkldnn_OIw8i16o2i,
        OIw8i8o = mkldnn_OIw8i8o,
        OIw8o16i2o = mkldnn_OIw8o16i2o,
        OIw8o8i = mkldnn_OIw8o8i,
        Owi16o = mkldnn_Owi16o,
        Owi4o = mkldnn_Owi4o,
        Owi8o = mkldnn_Owi8o,
        IOhw16o16i = mkldnn_IOhw16o16i,
        Ohwi16o = mkldnn_Ohwi16o,
        Ohwi4o = mkldnn_Ohwi4o,
        Ohwi8o = mkldnn_Ohwi8o,
        OIhw16i16o = mkldnn_OIhw16i16o,
        OIhw16o16i = mkldnn_OIhw16o16i,
        Oihw16o = mkldnn_Oihw16o,
        OIhw4i16o4i = mkldnn_OIhw4i16o4i,
        OIhw4i4o = mkldnn_OIhw4i4o,
        Oihw4o = mkldnn_Oihw4o,
        OIhw8i16o2i = mkldnn_OIhw8i16o2i,
        OIhw8i8o = mkldnn_OIhw8i8o,
        OIhw8o16i2o = mkldnn_OIhw8o16i2o,
        OIhw8o8i = mkldnn_OIhw8o8i,
        Odhwi16o = mkldnn_Odhwi16o,
        Odhwi4o = mkldnn_Odhwi4o,
        Odhwi8o = mkldnn_Odhwi8o,
        OIdhw16i16o = mkldnn_OIdhw16i16o,
        OIdhw16o16i = mkldnn_OIdhw16o16i,
        Oidhw16o = mkldnn_Oidhw16o,
        OIdhw4i4o = mkldnn_OIdhw4i4o,
        Oidhw4o = mkldnn_Oidhw4o,
        OIdhw8i16o2i = mkldnn_OIdhw8i16o2i,
        OIdhw8i8o = mkldnn_OIdhw8i8o,
        OIdhw8o8i = mkldnn_OIdhw8o8i,
        gIOw16o16i = mkldnn_gIOw16o16i,
        gOIw16i16o = mkldnn_gOIw16i16o,
        gOIw16o16i = mkldnn_gOIw16o16i,
        gOiw16o = mkldnn_gOiw16o,
        gOIw4i16o4i = mkldnn_gOIw4i16o4i,
        gOIw4i4o = mkldnn_gOIw4i4o,
        gOiw4o = mkldnn_gOiw4o,
        gOIw8i16o2i = mkldnn_gOIw8i16o2i,
        gOIw8i8o = mkldnn_gOIw8i8o,
        gOIw8o16i2o = mkldnn_gOIw8o16i2o,
        gOIw8o8i = mkldnn_gOIw8o8i,
        gOwi16o = mkldnn_gOwi16o,
        gOwi4o = mkldnn_gOwi4o,
        gOwi8o = mkldnn_gOwi8o,
        gIOhw16o16i = mkldnn_gIOhw16o16i,
        gOhwi16o = mkldnn_gOhwi16o,
        gOhwi4o = mkldnn_gOhwi4o,
        gOhwi8o = mkldnn_gOhwi8o,
        Goihw16g = mkldnn_Goihw16g,
        gOIhw16i16o = mkldnn_gOIhw16i16o,
        gOIhw16o16i = mkldnn_gOIhw16o16i,
        gOihw16o = mkldnn_gOihw16o,
        gOIhw2i8o4i = mkldnn_gOIhw2i8o4i,
        gOIhw4i16o4i = mkldnn_gOIhw4i16o4i,
        gOIhw4i4o = mkldnn_gOIhw4i4o,
        gOIhw4o4i = mkldnn_gOIhw4o4i,
        gOihw4o = mkldnn_gOihw4o,
        Goihw8g = mkldnn_Goihw8g,
        gOIhw8i16o2i = mkldnn_gOIhw8i16o2i,
        gOIhw8i8o = mkldnn_gOIhw8i8o,
        gOIhw8o16i2o = mkldnn_gOIhw8o16i2o,
        gOIhw8o8i = mkldnn_gOIhw8o8i,
        gOdhwi16o = mkldnn_gOdhwi16o,
        gOdhwi4o = mkldnn_gOdhwi4o,
        gOdhwi8o = mkldnn_gOdhwi8o,
        gOIdhw16i16o = mkldnn_gOIdhw16i16o,
        gOIdhw16o16i = mkldnn_gOIdhw16o16i,
        gOidhw16o = mkldnn_gOidhw16o,
        gOIdhw4i4o = mkldnn_gOIdhw4i4o,
        gOidhw4o = mkldnn_gOidhw4o,
        gOIdhw8i16o2i = mkldnn_gOIdhw8i16o2i,
        gOIdhw8i8o = mkldnn_gOIdhw8i8o,
        gOIdhw8o8i = mkldnn_gOIdhw8o8i,
    };

    /// A memory descriptor.
    struct desc {
        friend struct memory;
        /// The underlying C API data structure.
        mkldnn_memory_desc_t data;

        /// Constructs a zero memory descriptor
        desc(): data() {}

        /// Constructs a memory descriptor.
        ///
        /// @param adims Data dimensions
        /// @param adata_type Data precision/type.
        /// @param aformat Data layout format tag.
        desc(const dims &adims, data_type adata_type,
                format_tag aformat) {
            validate_dims(adims);
            error::wrap_c_api(mkldnn_memory_desc_init_by_tag(&data, (int)adims.size(),
                        adims.size() == 0 ? nullptr : &adims[0],
                        convert_to_c(adata_type), convert_to_c(aformat)),
                    "could not initialize a memory descriptor");
        }

        /// Constructs a memory descriptor from a C API data structure.
        ///
        /// @param adata A C API #mkldnn_memory_desc_t structure.
        desc(const mkldnn_memory_desc_t &adata): data(adata) {}

        /// Constructs a sub-memory descriptor
        //
        /// @param adims Sizes of a sub-memory
        /// @param offsets Offsets of a sub-memory
        desc submemory_desc(const dims &adims, const dims &offsets) {
            mkldnn_memory_desc_t sub_md;
            error::wrap_c_api(mkldnn_memory_desc_init_submemory(&sub_md,
                        &data, &adims[0], &offsets[0]),
                    "could not initialize a sub-memory");
            return desc(sub_md);
        }

        /// Returns the number of bytes required to allocate the memory described
        /// including the padding area.
        size_t get_size() const { return mkldnn_memory_desc_get_size(&data); }

        bool operator==(const desc &other) const {
            return mkldnn_memory_desc_equal(&data, &other.data) != 0;
        }

        bool operator!=(const desc &other) const { return !operator==(other); }
    };

    /// Constructs a memory.
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine.
    /// @param ahandle Native handle.
    memory(const desc &md, const engine &aengine, void *ahandle) {
        mkldnn_memory_t result;
        error::wrap_c_api(mkldnn_memory_create(&result, &md.data,
                    aengine.get(), ahandle), "could not create a memory");
        reset(result);
    }

    /// Constructs a memory.
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine.
    memory(const desc &md, const engine &aengine)
        : memory(md, aengine, MKLDNN_NATIVE_HANDLE_ALLOCATE) {}

    /// Returns the descriptor of the memory.
    desc get_desc() const {
        const mkldnn_memory_desc_t *cdesc;
        error::wrap_c_api(mkldnn_memory_get_memory_desc(get(), &cdesc),
                "could not get memory descriptor from a memory");
        return desc(*cdesc);
    }

    /// Returns the engine of the memory.
    engine get_engine() const {
        mkldnn_engine_t engine_q;
        error::wrap_c_api(mkldnn_memory_get_engine(get(), &engine_q),
                "could not get engine from a memory");
        return engine(engine_q);
    }

    /// Returns a handle of the data contained in the memory.
    ///
    /// On the CPU engine, this is a pointer to the allocated memory.
    void *get_data_handle() const {
        void *handle;
        error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle),
                "could not get native handle");
        return handle;
    }

    void set_data_handle(void *handle) const {
        error::wrap_c_api(mkldnn_memory_set_data_handle(get(), handle),
                "could not set native handle");
    }

    // Must go away or be private:
    static mkldnn_data_type_t convert_to_c(data_type adata_type) {
        return static_cast<mkldnn_data_type_t>(adata_type);
    }
    static mkldnn_format_tag_t convert_to_c(format_tag aformat) {
        return static_cast<mkldnn_format_tag_t>(aformat);
    }
};

inline bool operator==(mkldnn_data_type_t a, memory::data_type b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(mkldnn_data_type_t a, memory::data_type b) {
    return !(a == b);
}
inline bool operator==(memory::data_type a, mkldnn_data_type_t b) {
    return b == a;
}
inline bool operator!=(memory::data_type a, mkldnn_data_type_t b) {
    return !(a == b);
}

inline bool operator==(mkldnn_format_tag_t a, memory::format_tag b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(mkldnn_format_tag_t a, memory::format_tag b) {
    return !(a == b);
}
inline bool operator==(memory::format_tag a, mkldnn_format_tag_t b) {
    return b == a;
}
inline bool operator!=(memory::format_tag a, mkldnn_format_tag_t b) {
    return !(a == b);
}

/// @}

/// @addtogroup cpp_api_reorder Reorder
/// A primitive to copy data between memory formats.
///
/// @sa @ref c_api_reorder in @ref c_api
/// @{

struct reorder : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const engine &src_engine, const memory::desc &src_md,
                const engine &dst_engine, const memory::desc &dst_md,
                const primitive_attr &aattr) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_reorder_primitive_desc_create(&result,
                        src_engine.get(), &src_md.data,
                        dst_engine.get(), &dst_md.data, aattr.get()),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }

        primitive_desc(const engine &src_engine, const memory::desc &src_md,
                const engine &dst_engine, const memory::desc &dst_md) {
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_reorder_primitive_desc_create(&result,
                        src_engine.get(), &src_md.data,
                        dst_engine.get(), &dst_md.data, nullptr),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }

        primitive_desc(const memory &src, const memory &dst,
                const primitive_attr &aattr) {
            mkldnn_primitive_desc_t result;
            auto src_md = src.get_desc();
            auto dst_md = dst.get_desc();
            error::wrap_c_api(mkldnn_reorder_primitive_desc_create(&result,
                        src.get_engine().get(), &src_md.data,
                        dst.get_engine().get(), &dst_md.data, aattr.get()),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }

        primitive_desc(const memory &src, const memory &dst) {
            mkldnn_primitive_desc_t result;
            auto src_md = src.get_desc();
            auto dst_md = dst.get_desc();
            error::wrap_c_api(mkldnn_reorder_primitive_desc_create(&result,
                        src.get_engine().get(), &src_md.data,
                        dst.get_engine().get(), &dst_md.data, nullptr),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }

        memory::desc scratchpad_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(scratchpad_md), 0);
            if (cdesc == nullptr)
                return memory::desc();
            return memory::desc(*cdesc);
        }

        engine scratchpad_engine() {
            mkldnn_engine_t engine_q;
            error::wrap_c_api(
                mkldnn_primitive_desc_query(get(),
                    mkldnn::convert_to_c(query_scratchpad_engine), 0, &engine_q),
                "could not get scratchpad engine from reorder primitive_desc");

            return engine(engine_q);
        }

        engine get_engine() { return engine::query(*this); }
    };

    reorder(const primitive_desc &pd): primitive(pd.get()) {}

    reorder(const memory &src, const memory &dst):
        primitive(primitive_desc(src, dst).get()) {}

    void execute(stream astream, memory &src, memory &dst) {
        primitive::execute(astream,
                {{MKLDNN_ARG_FROM, src}, {MKLDNN_ARG_TO, dst}});
    }
};

/// @}

/// @addtogroup cpp_api_concat Concat
/// A primitive to concatenate data by arbitrary dimension.
///
/// @sa @ref c_api_concat in @ref c_api
/// @{

struct concat : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        std::vector<mkldnn_memory_desc_t> cpp_to_c(
                const std::vector<memory::desc> &srcs) {
            std::vector<mkldnn_memory_desc_t> c_api_srcs;
            c_api_srcs.reserve(srcs.size());
            for (const auto &s : srcs) c_api_srcs.push_back(s.data);
            return c_api_srcs;
        }

        primitive_desc(const memory::desc &dst, int concat_dimension,
                const std::vector<memory::desc> &srcs, const engine &aengine) {
            auto c_api_srcs = cpp_to_c(srcs);

            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_concat_primitive_desc_create(
                    &result, &dst.data, (int)c_api_srcs.size(),
                    concat_dimension, &c_api_srcs[0], nullptr, aengine.get()),
                "could not create a concat primitive descriptor");
            reset(result);
        }

        primitive_desc(int concat_dimension,
                const std::vector<memory::desc> &srcs, const engine &aengine) {
            auto c_api_srcs = cpp_to_c(srcs);

            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_concat_primitive_desc_create(
                    &result, nullptr, (int)c_api_srcs.size(),
                    concat_dimension, &c_api_srcs[0], nullptr, aengine.get()),
                "could not create a concat primitive descriptor");
            reset(result);
        }

        memory::desc dst_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(dst_md), 0);
            error::wrap_c_api(
                    cdesc == nullptr ? mkldnn_runtime_error : mkldnn_success,
                    "could not get a dst memory descriptor");
            return memory::desc(*cdesc);
        }

        memory::desc scratchpad_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(scratchpad_md), 0);
            if (cdesc == nullptr)
                return memory::desc();
            return memory::desc(*cdesc);
        }

        engine get_engine() { return engine::query(*this); }
    };

    concat(const primitive_desc &pd): primitive(pd.get()) {}
};

/// @}

/// @addtogroup cpp_api_sum Sum
/// A primitive to sum data.
///
/// @sa @ref c_api_sum in @ref c_api
/// @{

struct sum : public primitive {
    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        std::vector<mkldnn_memory_desc_t> cpp_to_c(
                const std::vector<memory::desc> &srcs) {
            std::vector<mkldnn_memory_desc_t> c_api_srcs;
            c_api_srcs.reserve(srcs.size());
            for (const auto &s : srcs) c_api_srcs.push_back(s.data);
            return c_api_srcs;
        }

        primitive_desc(const memory::desc &dst,
                const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs, const engine &aengine) {
            error::wrap_c_api(scales.size() == srcs.size()
                    ? mkldnn_success : mkldnn_invalid_arguments,
                "number of scales not equal to number of srcs");

            auto c_api_srcs = cpp_to_c(srcs);

            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_sum_primitive_desc_create(
                    &result, &dst.data, (int)c_api_srcs.size(),
                    &scales[0], &c_api_srcs[0], nullptr, aengine.get()),
                "could not create a sum primitive descriptor");
            reset(result);
        }

        primitive_desc(const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs, const engine &aengine) {
            error::wrap_c_api(scales.size() == srcs.size()
                    ? mkldnn_success : mkldnn_invalid_arguments,
                "number of scales not equal to number of srcs");

            auto c_api_srcs = cpp_to_c(srcs);
            mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_sum_primitive_desc_create(&result,
                        nullptr, (int)c_api_srcs.size(), &scales[0],
                        &c_api_srcs[0], nullptr, aengine.get()),
                    "could not create a sum primitive descriptor");
            reset(result);
        }

        memory::desc dst_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(dst_md), 0);
            error::wrap_c_api(
                    cdesc == nullptr ? mkldnn_runtime_error : mkldnn_success,
                    "could not get a dst memory descriptor");
            return memory::desc(*cdesc);
        }

        memory::desc scratchpad_desc() const {
            const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                    get(), mkldnn::convert_to_c(scratchpad_md), 0);
            if (cdesc == nullptr)
                return memory::desc();
            return memory::desc(*cdesc);
        }

        engine get_engine() { return engine::query(*this); }
    };

    sum(const primitive_desc &pd): primitive(pd.get()) {}
};

/// @}

/// @}

/// @addtogroup cpp_api_primitives Primitives
/// @{

/// @addtogroup cpp_api_primitive_descriptors Primitive descriptors
/// @{

/// A base class for all primitive descriptors.
struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
    primitive_desc(const_mkldnn_op_desc_t desc, const primitive_attr *attr,
            const engine &e, const_mkldnn_primitive_desc_t hint_fwd_pd) {
        mkldnn_primitive_desc_iterator_t iterator = nullptr;
        mkldnn_status_t status = mkldnn_primitive_desc_iterator_create(
                &iterator, desc, attr ? attr->get() : nullptr, e.get(),
                hint_fwd_pd);
        error::wrap_c_api(status,
                "could not create a primitive descriptor iterator");
        pd_iterator.reset(iterator);
        fetch_impl();
    }

    engine get_engine() { return engine::query(*this); }

    primitive_attr get_primitive_attr() const {
        const_mkldnn_primitive_attr_t const_cattr;
        error::wrap_c_api(mkldnn_primitive_desc_get_attr(get(), &const_cattr),
                "could not get attributes");
        mkldnn_primitive_attr_t cattr;
        error::wrap_c_api(mkldnn_primitive_attr_clone(&cattr, const_cattr),
                "could not clone attributes");

        primitive_attr attr;
        attr.reset(cattr);
        return attr;
    }

    /// Returns implementation name
    const char *impl_info_str() const {
        const char *res;
        error::wrap_c_api(mkldnn_primitive_desc_query(get(),
                    mkldnn_query_impl_info_str, 0, &res),
                "could not query implementation info string");
        return res;
    }

    /// Queries the memory::dim value (same as int64_t)
    memory::dim query_s64(query q) const {
        memory::dim res;
        mkldnn_status_t status = mkldnn_primitive_desc_query(get(),
                mkldnn::convert_to_c(q), 0, &res);
        return status == mkldnn_success ? res : 0;
    }

    /// Advances the next implementation for the given op descriptor.
    ///
    /// Returns:
    /// - @c true on success
    /// - @c false if the last implementation reached, and
    ///   the primitive descriptor itself is kept unchanged
    bool next_impl() {
        mkldnn_status_t status = mkldnn_primitive_desc_iterator_next(
                pd_iterator.get());
        if (status == mkldnn_iterator_ends) return false;
        error::wrap_c_api(status, "primitive descriptor iterator next failed");

        fetch_impl();
        return true;
    }

    /// Queries and returns requested memory descriptor.
    memory::desc query_md(query what, int idx = 0) const {
        std::vector<query> valid_q{src_md, diff_src_md, weights_md,
            diff_weights_md, dst_md, diff_dst_md, workspace_md, scratchpad_md};
        if (!std::any_of(valid_q.cbegin(), valid_q.cend(),
                    [=](query q) { return what == q; }))
            throw error(mkldnn_invalid_arguments, "invalid memory query");

        const mkldnn_memory_desc_t *cdesc = mkldnn_primitive_desc_query_md(
                get(), mkldnn::convert_to_c(what), idx);
        if (cdesc == nullptr) return memory::desc();

        return memory::desc(*cdesc);
    }

    // register specialized queries, e.g. src_desc()
#   define REG_QUERY_MD(name, what, idx) \
    memory::desc name ## _desc() const { return query_md(what ## _md, idx); }

  private:
    handle<mkldnn_primitive_desc_iterator_t> pd_iterator;
    void fetch_impl() {
        mkldnn_primitive_desc_t pd = mkldnn_primitive_desc_iterator_fetch(
                pd_iterator.get());
        error::wrap_c_api(pd != nullptr ? mkldnn_success : mkldnn_runtime_error,
                "could not fetch a primitive descriptor from the iterator");
        reset(pd);
    }
};

/// @}

/// @addtogroup cpp_api_convolution Convolution
/// A primitive to compute convolution using different algorithms.
///
/// @sa @ref c_api_convolution in @ref c_api
/// @{

struct convolution_forward: public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                mkldnn_dilated_convolution_forward_desc_init(&data,
                    mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &dilates[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated convolution forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                mkldnn_dilated_convolution_forward_desc_init(&data,
                    mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &dilates[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated convolution forward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(bias, weights, 1);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    convolution_forward(const primitive_desc &pd): primitive(pd) {}
};

struct convolution_backward_data : public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_data_desc_init(
                        &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward data descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                mkldnn_dilated_convolution_backward_data_desc_init(
                    &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                    &weights_desc.data, &diff_dst_desc.data,
                    &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                    mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward data descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const convolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const convolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    convolution_backward_data(const primitive_desc &pd): primitive(pd) {}
};

struct convolution_backward_weights : public primitive {
    struct desc {
        mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &dilates[0],  &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }

    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const convolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const convolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(diff_weights, diff_weights, 0);
        REG_QUERY_MD(diff_bias, diff_weights, 1);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    convolution_backward_weights(const primitive_desc &pd): primitive(pd) {}
};

/// @}
//
/// @addtogroup cpp_api_deconvolution Deconvolution
/// A primitive to compute deconvolution using different algorithms.
///
/// @sa @ref c_api_deconvolution in @ref c_api
/// @{

struct deconvolution_forward: public primitive {
    struct desc {
        mkldnn_deconvolution_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &dilates[0], &padding_l[0],
                        &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated deconvolution forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &dilates[0], &padding_l[0],
                        &padding_r[0], mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated deconvolution forward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(bias, weights, 1);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    deconvolution_forward(const primitive_desc &pd): primitive(pd) {}
};

struct deconvolution_backward_data : public primitive {
    struct desc {
        mkldnn_deconvolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_backward_data_desc_init(
                        &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution backward data descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_backward_data_desc_init(
                        &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated deconvolution backward data descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const deconvolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const deconvolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    deconvolution_backward_data(const primitive_desc &pd): primitive(pd) {}
};

struct deconvolution_backward_weights : public primitive {
    struct desc {
        mkldnn_deconvolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_deconvolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a deconvolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated  deconvolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims dilates,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_dilated_deconvolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a dilated deconvolution backward weights descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const deconvolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const deconvolution_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(diff_weights, diff_weights, 0);
        REG_QUERY_MD(diff_bias, diff_weights, 1);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    deconvolution_backward_weights(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_lrn LRN
/// A primitive to perform local response normalization (LRN) across or within
/// channels.
///
/// @sa @ref c_api_lrn in @ref c_api
/// @{

struct lrn_forward : public primitive {
    struct desc {
        mkldnn_lrn_desc_t data;

        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, memory::dim local_size,
                float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(mkldnn_lrn_forward_desc_init(&data,
                mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, local_size, alpha, beta, k),
                "could not create a lrn forward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    lrn_forward(const primitive_desc &pd): primitive(pd) {}
};

struct lrn_backward : public primitive {
    struct desc {
        mkldnn_lrn_desc_t data;

        desc(algorithm aalgorithm, const memory::desc &data_desc,
                const memory::desc &diff_data_desc, memory::dim local_size,
                float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(mkldnn_lrn_backward_desc_init(&data,
                convert_to_c(aalgorithm), &diff_data_desc.data,
                &data_desc.data, local_size, alpha, beta, k),
                "could not create a lrn backward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const lrn_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const lrn_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    lrn_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_pooling Pooling
/// A primitive to perform max or average pooling.
///
/// @sa @ref c_api_pooling in @ref c_api
/// @{

struct pooling_forward : public primitive {
    struct desc {
        mkldnn_pooling_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims kernel,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_pooling_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm),
                        &src_desc.data, &dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a forward pooling descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    pooling_forward(const primitive_desc &pd): primitive(pd) {}
};

struct pooling_backward : public primitive {
    struct desc {
        mkldnn_pooling_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims &strides,
                const memory::dims &kernel,
                const memory::dims &padding_l,
                const memory::dims &padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_pooling_backward_desc_init(&data,
                        convert_to_c(aalgorithm),
                        &diff_src_desc.data, &diff_dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a backward pooling descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const pooling_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const pooling_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    pooling_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_eltwise Eltwise
/// A primitive to compute element-wise operations like parametric rectifier
/// linear unit (ReLU).
///
/// @sa @ref c_api_eltwise in @ref c_api
/// @{

struct eltwise_forward : public primitive {
    struct desc {
        mkldnn_eltwise_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, algorithm alg_kind,
                const memory::desc &src_desc, T alpha = 0, T beta = 0) {
            error::wrap_c_api(mkldnn_eltwise_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        mkldnn::convert_to_c(alg_kind), &src_desc.data,
                        static_cast<float>(alpha), static_cast<float>(beta)),
                    "could not create a eltwise forward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    eltwise_forward(const primitive_desc &pd): primitive(pd) {}
};

struct eltwise_backward : public primitive {
    struct desc {
        mkldnn_eltwise_desc_t data;

        template <typename T>
        desc(algorithm alg_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, T alpha = 0, T beta = 0) {
            error::wrap_c_api(mkldnn_eltwise_backward_desc_init(&data,
                        mkldnn::convert_to_c(alg_kind), &diff_data_desc.data,
                        &data_desc.data, static_cast<float>(alpha),
                        static_cast<float>(beta)),
                    "could not create a eltwise backward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const eltwise_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const eltwise_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    eltwise_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_softmax Softmax
/// A primitive to perform softmax.
///
/// @sa @ref c_api_softmax in @ref c_api
/// @{

struct softmax_forward : public primitive {
    struct desc {
        mkldnn_softmax_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             int softmax_axis) {
            error::wrap_c_api(mkldnn_softmax_forward_desc_init(&data,
                    mkldnn::convert_to_c(aprop_kind), &data_desc.data,
                    softmax_axis),
                "could not create a softmax forward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    softmax_forward(const primitive_desc &pd): primitive(pd) {}
};

struct softmax_backward : public primitive {
    struct desc {
        mkldnn_softmax_desc_t data;
        desc(const memory::desc &diff_desc, const memory::desc &data_desc,
                int softmax_axis) {
            error::wrap_c_api(mkldnn_softmax_backward_desc_init(&data,
                        &diff_desc.data, &data_desc.data, softmax_axis),
                    "could not init a backward softmax descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const softmax_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const softmax_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    softmax_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_batch_norm Batch normalization
/// A primitive to perform batch normalization.
///
/// @sa @ref c_api_batch_normalization in @ref c_api
/// @{

struct batch_normalization_forward : public primitive {
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &src_desc, T epsilon,
                unsigned flags) {
            error::wrap_c_api(
                    mkldnn_batch_normalization_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        static_cast<float>(epsilon), flags),
                "could not create a batch normalization forward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);

        memory::desc mean_desc() const { return stat_desc(mean); }
        memory::desc variance_desc() const { return stat_desc(var); }

    private:
        enum { mean = 1, var = 2, };
        memory::desc stat_desc(int kind) const {
            mkldnn_batch_normalization_desc_t *p;
            error::wrap_c_api(mkldnn_primitive_desc_query(
                    get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                    "could not get a batch-normalization descriptor");
            return query_md(p->flags & use_global_stats ? src_md : dst_md, kind);
        }
    };

    batch_normalization_forward(const primitive_desc &pd): primitive(pd) {}
};

struct batch_normalization_backward : public primitive {
    struct desc {
        mkldnn_batch_normalization_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, T epsilon, unsigned flags) {
            error::wrap_c_api(
                    mkldnn_batch_normalization_backward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        &diff_data_desc.data, &data_desc.data,
                        static_cast<float>(epsilon), flags),
                "could not create a batch normalization backward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(mean, src, 1);
        REG_QUERY_MD(variance, src, 2);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(workspace, workspace, 0);

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_weights, diff_weights, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    batch_normalization_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_inner_product Inner Product
/// A primitive to compute an inner product.
///
/// @sa @ref c_api_inner_product in @ref c_api
/// @{

struct inner_product_forward: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, &bias_desc.data, &dst_desc.data),
                    "could not create a inner product forward descriptor");
        }

        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, nullptr, &dst_desc.data),
                    "could not create a inner product forward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(bias, weights, 1);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    inner_product_forward(const primitive_desc &pd): primitive(pd) {}
};

struct inner_product_backward_data: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_data_desc_init(&data,
                        &diff_src_desc.data, &weights_desc.data,
                        &diff_dst_desc.data),
                "could not create a inner product backward data descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const inner_product_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const inner_product_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(weights, weights, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    inner_product_backward_data(const primitive_desc &pd): primitive(pd) {}
};

struct inner_product_backward_weights: public primitive {
    struct desc {
        mkldnn_inner_product_desc_t data;
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_weights_desc_init(
                        &data, &src_desc.data, &diff_weights_desc.data,
                        &diff_bias_desc.data, &diff_dst_desc.data),
                "could not create a inner product backward weights descriptor");
        }
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    mkldnn_inner_product_backward_weights_desc_init(
                        &data, &src_desc.data, &diff_weights_desc.data,
                        nullptr, &diff_dst_desc.data),
                "could not create a inner product backward weights descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const inner_product_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const inner_product_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(diff_weights, diff_weights, 0);
        REG_QUERY_MD(diff_bias, diff_weights, 1);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    inner_product_backward_weights(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_rnn RNN
/// A primitive to compute common recurrent layer.
///
/// @sa @ref c_api_rnn in @ref c_api
/// @{

struct rnn_cell {
    struct desc {
        mkldnn_rnn_cell_desc_t c_rnn_cell_;

        desc(algorithm kind, algorithm activation_f) {
            error::wrap_c_api(mkldnn_rnn_cell_desc_init(&c_rnn_cell_,
                        mkldnn::convert_to_c(kind),
                        mkldnn::convert_to_c(activation_f), 0U, 0, 0),
                    "could not init an rnn cell descriptor");
        }
        desc(algorithm kind): desc(kind, algorithm::algorithm_undef) {}

        operator const mkldnn_rnn_cell_desc_t*() const { return &c_rnn_cell_; }

        algorithm get_cell_kind() const
        { return algorithm(c_rnn_cell_.cell_kind); }
        algorithm get_activation() const
        { return algorithm(c_rnn_cell_.activation_kind); }

        float get_alpha() const { return c_rnn_cell_.alpha; }
        void set_alpha(float alpha) {
            c_rnn_cell_.flags |= mkldnn_rnn_cell_with_relu;
            c_rnn_cell_.alpha = alpha;
        }

        float get_clipping() const { return c_rnn_cell_.clipping; }
        void set_clipping(float clipping) {
            c_rnn_cell_.flags |= mkldnn_rnn_cell_with_clipping;
            c_rnn_cell_.clipping = clipping;
        }

        int get_gates_count() const {
            return mkldnn_rnn_cell_get_gates_count(&c_rnn_cell_);
        }
        int get_state_count() const {
            return mkldnn_rnn_cell_get_states_count(&c_rnn_cell_);
        }
    };
};

struct rnn_forward : public primitive {
    struct desc {
        mkldnn_rnn_desc_t data;
        desc(prop_kind aprop_kind, rnn_cell::desc cell,
                const rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc
            ) {
            error::wrap_c_api(mkldnn_rnn_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), cell,
                        mkldnn::convert_to_c(direction),
                        &src_layer_desc.data, &src_iter_desc.data,
                        &weights_layer_desc.data, &weights_iter_desc.data,
                        &bias_desc.data,
                        &dst_layer_desc.data, &dst_iter_desc.data),
                    "could not create an RNN forward descriptor");
        }

    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        REG_QUERY_MD(src_layer, src, 0);
        REG_QUERY_MD(src_iter, src, 1);
        REG_QUERY_MD(weights_layer, weights, 0);
        REG_QUERY_MD(weights_iter, weights, 1);
        REG_QUERY_MD(bias, weights, 2);
        REG_QUERY_MD(dst_layer, dst, 0);
        REG_QUERY_MD(dst_iter, dst, 1);
        REG_QUERY_MD(workspace, workspace, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    rnn_forward(const primitive_desc &pd): primitive(pd) {}
};

struct rnn_backward : public primitive {
    struct desc {
        mkldnn_rnn_desc_t data;
        desc(prop_kind aprop_kind, rnn_cell::desc cell,
                const rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc) {
            error::wrap_c_api(mkldnn_rnn_backward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), cell,
                        mkldnn::convert_to_c(direction),
                        &src_layer_desc.data, &src_iter_desc.data,
                        &weights_layer_desc.data, &weights_iter_desc.data,
                        &bias_desc.data,
                        &dst_layer_desc.data, &dst_iter_desc.data,
                        &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                        &diff_weights_layer_desc.data,
                        &diff_weights_iter_desc.data, &diff_bias_desc.data,
                        &diff_dst_layer_desc.data, &diff_dst_iter_desc.data),
                    "could not create an RNN backward descriptor");
        }

    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const rnn_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e,
                const rnn_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, &attr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(src_layer, src, 0);
        REG_QUERY_MD(src_iter, src, 1);
        REG_QUERY_MD(weights_layer, weights, 0);
        REG_QUERY_MD(weights_iter, weights, 1);
        REG_QUERY_MD(bias, weights, 2);
        REG_QUERY_MD(dst_layer, dst, 0);
        REG_QUERY_MD(dst_iter, dst, 1);
        REG_QUERY_MD(workspace, workspace, 0);

        REG_QUERY_MD(diff_src_layer, diff_src, 0);
        REG_QUERY_MD(diff_src_iter, diff_src, 1);
        REG_QUERY_MD(diff_weights_layer, diff_weights, 0);
        REG_QUERY_MD(diff_weights_iter, diff_weights, 1);
        REG_QUERY_MD(diff_bias, diff_weights, 2);
        REG_QUERY_MD(diff_dst_layer, diff_dst, 0);
        REG_QUERY_MD(diff_dst_iter, diff_dst, 1);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    // With last iteration (with and without input src_iter)
    rnn_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_shuffle Shuffle
/// A primitive to shuffle data along the axis.
///
/// @sa @ref c_api_shuffle in @ref c_api
/// @{

struct shuffle_forward : public primitive {
    struct desc {
        mkldnn_shuffle_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
                int axis, int group_size) {
            error::wrap_c_api(mkldnn_shuffle_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &data_desc.data,
                        axis, group_size),
                    "could not create a shuffle forward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        REG_QUERY_MD(src, src, 0);
        REG_QUERY_MD(dst, dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    shuffle_forward(const primitive_desc &pd): primitive(pd) {}
};

struct shuffle_backward : public primitive {
    struct desc {
        mkldnn_shuffle_desc_t data;
        desc(const memory::desc &diff_data_desc, int axis, int group_size) {
            error::wrap_c_api(mkldnn_shuffle_backward_desc_init(&data,
                        &diff_data_desc.data, axis, group_size),
                    "could not create a shuffle backward descriptor");
        }
    };

    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc(const desc &desc, const engine &e,
                const shuffle_forward::primitive_desc &hint_fwd_pd)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, hint_fwd_pd.get()) {}

        REG_QUERY_MD(diff_src, diff_src, 0);
        REG_QUERY_MD(diff_dst, diff_dst, 0);
        REG_QUERY_MD(scratchpad, scratchpad, 0);
    };

    shuffle_backward(const primitive_desc &pd): primitive(pd) {}
};

/// @}

/// @} Primitives

/// @} C++ API

#undef REG_QUERY_MD

// implementation section
#ifndef DOXYGEN_SHOULD_SKIP_THIS

inline primitive::primitive(const_mkldnn_primitive_desc_t c_pd) {
    mkldnn_primitive_t result;
    error::wrap_c_api(mkldnn_primitive_create(&result, c_pd),
            "could not create a primitive");
    reset(result);
}

inline primitive::primitive(const primitive_desc &pd): primitive(pd.get()) {}

inline void primitive::execute(stream &astream,
        const std::unordered_map<int, memory> &args) const {
    std::vector<mkldnn_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a: args)
        c_args.push_back({a.first, a.second.get()});

    error::wrap_c_api(mkldnn_primitive_execute(get(), astream.get(),
                (int)c_args.size(), c_args.data()),
            "primitive execution fail");
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace mkldnn

#endif
