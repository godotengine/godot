#include "projection_validator.h"

#include "core/io/dir_access.h"
#include "core/io/image.h"
#include "core/io/json.h"
#include "core/math/color.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

namespace GaussianProjectionValidation {

namespace {

template <typename T>
static T clamp_min(const T &p_value, const T &p_minimum) {
    return p_value < p_minimum ? p_minimum : p_value;
}

} // namespace

struct ProjectionValidator::Matrix3f {
    float m[3][3];

    Matrix3f() {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                m[r][c] = (r == c) ? 1.0f : 0.0f;
            }
        }
    }

    static Matrix3f zero() {
        Matrix3f result;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                result.m[r][c] = 0.0f;
            }
        }
        return result;
    }
};

ProjectionValidator::ProjectionValidator() {}

ProjectionValidator::~ProjectionValidator() {
    shutdown();
}

Error ProjectionValidator::initialize(RenderingDevice *p_rd) {
    ERR_FAIL_COND_V(p_rd == nullptr, ERR_INVALID_PARAMETER);
    if (rd && rd != p_rd) {
        shutdown();
    }
    rd = p_rd;
    return ensure_shader();
}

void ProjectionValidator::shutdown() {
    if (rd) {
        if (pipeline.is_valid()) {
            rd->free(pipeline);
        }
        if (shader.is_valid()) {
            rd->free(shader);
        }
    }
    pipeline = RID();
    shader = RID();
    rd = nullptr;
}

ProjectionValidator::Matrix3f ProjectionValidator::basis_to_matrix(const Basis &p_basis) const {
    Matrix3f m = Matrix3f::zero();
    for (int r = 0; r < 3; ++r) {
        const Vector3 &row = p_basis[r];
        m.m[r][0] = (float)row.x;
        m.m[r][1] = (float)row.y;
        m.m[r][2] = (float)row.z;
    }
    return m;
}

ProjectionValidator::Matrix3f ProjectionValidator::build_rotation_matrix(const Quaternion &p_rotation) const {
    Quaternion q = p_rotation;
    if (!q.is_normalized()) {
        q.normalize();
    }

    const float xx = (float)(q.x * q.x);
    const float yy = (float)(q.y * q.y);
    const float zz = (float)(q.z * q.z);
    const float xy = (float)(q.x * q.y);
    const float xz = (float)(q.x * q.z);
    const float yz = (float)(q.y * q.z);
    const float wx = (float)(q.w * q.x);
    const float wy = (float)(q.w * q.y);
    const float wz = (float)(q.w * q.z);

    Matrix3f result = Matrix3f::zero();
    result.m[0][0] = 1.0f - 2.0f * (yy + zz);
    result.m[0][1] = 2.0f * (xy - wz);
    result.m[0][2] = 2.0f * (xz + wy);

    result.m[1][0] = 2.0f * (xy + wz);
    result.m[1][1] = 1.0f - 2.0f * (xx + zz);
    result.m[1][2] = 2.0f * (yz - wx);

    result.m[2][0] = 2.0f * (xz - wy);
    result.m[2][1] = 2.0f * (yz + wx);
    result.m[2][2] = 1.0f - 2.0f * (xx + yy);

    return result;
}

ProjectionValidator::Matrix3f ProjectionValidator::multiply(const Matrix3f &p_a, const Matrix3f &p_b) const {
    Matrix3f result = Matrix3f::zero();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float value = 0.0f;
            for (int k = 0; k < 3; ++k) {
                value += p_a.m[r][k] * p_b.m[k][c];
            }
            result.m[r][c] = value;
        }
    }
    return result;
}

ProjectionValidator::Matrix3f ProjectionValidator::transpose(const Matrix3f &p_m) const {
    Matrix3f result = Matrix3f::zero();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            result.m[r][c] = p_m.m[c][r];
        }
    }
    return result;
}

ProjectionValidator::Matrix3f ProjectionValidator::build_covariance_matrix(const Quaternion &p_rotation, const Vector3 &p_scale) const {
    Matrix3f rotation_matrix = build_rotation_matrix(p_rotation);

    Matrix3f scale_matrix = Matrix3f::zero();
    scale_matrix.m[0][0] = (float)(p_scale.x * p_scale.x);
    scale_matrix.m[1][1] = (float)(p_scale.y * p_scale.y);
    scale_matrix.m[2][2] = (float)(p_scale.z * p_scale.z);

    Matrix3f temp = multiply(rotation_matrix, scale_matrix);
    Matrix3f covariance = multiply(temp, transpose(rotation_matrix));
    return covariance;
}

ProjectedGaussian ProjectionValidator::project_from_matrices(const Matrix3f &p_view, const Matrix3f &p_covariance) const {
    Matrix3f view_times_cov = multiply(p_view, p_covariance);
    Matrix3f covariance_2d = multiply(view_times_cov, transpose(p_view));

    float cov_xx = covariance_2d.m[0][0];
    float cov_xy = covariance_2d.m[0][1];
    float cov_yy = covariance_2d.m[1][1];

    cov_xx += variance_epsilon;
    cov_yy += variance_epsilon;

    cov_xx = clamp_min(cov_xx, min_variance);
    cov_yy = clamp_min(cov_yy, min_variance);

    float determinant = cov_xx * cov_yy - cov_xy * cov_xy;
    float safe_determinant = clamp_min(determinant, min_determinant);
    float inv_det = 1.0f / safe_determinant;

    ProjectedGaussian result;
    result.cov_xx = cov_xx;
    result.cov_yy = cov_yy;
    result.cov_xy = cov_xy;
    result.determinant = determinant;
    result.conic = Vector3(cov_yy * inv_det, -cov_xy * inv_det, cov_xx * inv_det);
    return result;
}

ProjectedGaussian ProjectionValidator::project_cpu(const ProjectionInput &p_input) const {
    return project_cpu(p_input.view_basis, p_input.rotation, p_input.scale);
}

ProjectedGaussian ProjectionValidator::project_cpu(const Basis &p_view_basis, const Quaternion &p_rotation, const Vector3 &p_scale) const {
    Matrix3f view_matrix = basis_to_matrix(p_view_basis);
    Matrix3f covariance_matrix = build_covariance_matrix(p_rotation, p_scale);
    return project_from_matrices(view_matrix, covariance_matrix);
}

Error ProjectionValidator::project_gpu(const Vector<ProjectionInput> &p_inputs, Vector<ProjectedGaussian> &r_results) {
    ERR_FAIL_COND_V(p_inputs.is_empty(), ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(rd == nullptr, ERR_UNCONFIGURED);

    Error err = ensure_shader();
    ERR_FAIL_COND_V(err != OK, err);

    struct GPUInput {
        float view_col0[4];
        float view_col1[4];
        float view_col2[4];
        float cov_col0[4];
        float cov_col1[4];
        float cov_col2[4];
    };

    struct GPUOutput {
        float conic_det[4];
        float covariance[4];
    };

    Vector<GPUInput> input_data;
    ERR_FAIL_COND_V(input_data.resize(p_inputs.size()) != OK, ERR_OUT_OF_MEMORY);

    for (int i = 0; i < p_inputs.size(); ++i) {
        const ProjectionInput &input = p_inputs[i];
        Matrix3f view_matrix = basis_to_matrix(input.view_basis);
        Matrix3f covariance_matrix = build_covariance_matrix(input.rotation, input.scale);

        GPUInput packed = {};
        for (int c = 0; c < 3; ++c) {
            packed.view_col0[c] = view_matrix.m[c][0];
            packed.view_col1[c] = view_matrix.m[c][1];
            packed.view_col2[c] = view_matrix.m[c][2];
            packed.cov_col0[c] = covariance_matrix.m[c][0];
            packed.cov_col1[c] = covariance_matrix.m[c][1];
            packed.cov_col2[c] = covariance_matrix.m[c][2];
        }
        packed.view_col0[3] = 0.0f;
        packed.view_col1[3] = 0.0f;
        packed.view_col2[3] = 0.0f;
        packed.cov_col0[3] = 0.0f;
        packed.cov_col1[3] = 0.0f;
        packed.cov_col2[3] = 0.0f;

        input_data.write[i] = packed;
    }

    RID input_buffer = rd->storage_buffer_create(input_data.size() * sizeof(GPUInput), Span<uint8_t>((uint8_t *)input_data.ptr(), input_data.size() * sizeof(GPUInput)));
    ERR_FAIL_COND_V(!input_buffer.is_valid(), ERR_CANT_CREATE);

    RID output_buffer = rd->storage_buffer_create(p_inputs.size() * sizeof(GPUOutput));
    if (!output_buffer.is_valid()) {
        rd->free(input_buffer);
        ERR_FAIL_V(ERR_CANT_CREATE);
    }

    Vector<RD::Uniform> uniforms;
    ERR_FAIL_COND_V(uniforms.resize(2) != OK, ERR_OUT_OF_MEMORY);

    uniforms.write[0].uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    uniforms.write[0].binding = 0;
    uniforms.write[0].append_id(input_buffer);

    uniforms.write[1].uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    uniforms.write[1].binding = 1;
    uniforms.write[1].append_id(output_buffer);

    RID uniform_set = rd->uniform_set_create(uniforms, shader, 0);
    if (!uniform_set.is_valid()) {
        rd->free(input_buffer);
        rd->free(output_buffer);
        ERR_FAIL_V(ERR_CANT_CREATE);
    }

    uint32_t case_count = p_inputs.size();
    uint32_t local_size = 64;
    uint32_t groups = (case_count + local_size - 1) / local_size;

    RD::ComputeListID compute_list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(compute_list, pipeline);
    rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
    rd->compute_list_set_push_constant(compute_list, &case_count, sizeof(uint32_t));
    rd->compute_list_dispatch(compute_list, groups, 1, 1);
    rd->compute_list_end();

    rd->sync();

    Vector<uint8_t> raw_output = rd->buffer_get_data(output_buffer);
    const GPUOutput *gpu_data = reinterpret_cast<const GPUOutput *>(raw_output.ptr());

    r_results.clear();
    ERR_FAIL_COND_V(r_results.resize(p_inputs.size()) != OK, ERR_OUT_OF_MEMORY);
    for (int i = 0; i < p_inputs.size(); ++i) {
        ProjectedGaussian result;
        result.conic = Vector3(gpu_data[i].conic_det[0], gpu_data[i].conic_det[1], gpu_data[i].conic_det[2]);
        result.determinant = gpu_data[i].conic_det[3];
        result.cov_xx = gpu_data[i].covariance[0];
        result.cov_yy = gpu_data[i].covariance[1];
        result.cov_xy = gpu_data[i].covariance[2];
        r_results.write[i] = result;
    }

    rd->free(uniform_set);
    rd->free(input_buffer);
    rd->free(output_buffer);

    return OK;
}

void ProjectionValidator::compute_eigenvalues(const ProjectedGaussian &p_result, double &r_lambda_min, double &r_lambda_max) const {
    double trace = (double)p_result.cov_xx + (double)p_result.cov_yy;
    double det = (double)p_result.cov_xx * (double)p_result.cov_yy - (double)p_result.cov_xy * (double)p_result.cov_xy;
    double half_trace = trace * 0.5;
    double discriminant = half_trace * half_trace - det;
    discriminant = MAX(discriminant, 0.0);
    double root = Math::sqrt(discriminant);
    r_lambda_max = half_trace + root;
    r_lambda_min = half_trace - root;
}

double ProjectionValidator::compute_positive_semidefinite_margin(const ProjectedGaussian &p_result) const {
    double lambda_min = 0.0;
    double lambda_max = 0.0;
    compute_eigenvalues(p_result, lambda_min, lambda_max);
    return lambda_min;
}

double ProjectionValidator::compute_scale_invariance_error(const ProjectionInput &p_input, float p_uniform_scale) const {
    ProjectionInput scaled = p_input;
    scaled.scale *= p_uniform_scale;

    ProjectedGaussian base = project_cpu(p_input);
    ProjectedGaussian scaled_result = project_cpu(scaled);

    float scale_sq = p_uniform_scale * p_uniform_scale;

    auto relative = [](float expected, float actual) {
        if (Math::is_zero_approx(expected)) {
            return (double)Math::abs(actual);
        }
        return (double)Math::abs((actual - expected) / expected);
    };

    float base_x = base.cov_xx - variance_epsilon;
    float base_y = base.cov_yy - variance_epsilon;
    double err_x = relative(base_x * scale_sq, scaled_result.cov_xx - variance_epsilon);
    double err_y = relative(base_y * scale_sq, scaled_result.cov_yy - variance_epsilon);
    double err_xy = relative(base.cov_xy * scale_sq, scaled_result.cov_xy);
    return MAX(err_x, MAX(err_y, err_xy));
}

double ProjectionValidator::compute_rotation_equivariance_error(const ProjectionInput &p_input, const Basis &p_rotation) const {
    Basis additional_rotation = p_rotation;
    Basis gaussian_basis;
    gaussian_basis.set_quaternion(p_input.rotation);

    Basis rotated_gaussian_basis = additional_rotation * gaussian_basis;
    Basis rotated_view = p_input.view_basis * additional_rotation.transposed();

    ProjectionInput rotated = p_input;
    rotated.view_basis = rotated_view;
    rotated.rotation = rotated_gaussian_basis.get_quaternion();

    ProjectedGaussian base = project_cpu(p_input);
    ProjectedGaussian rotated_result = project_cpu(rotated);

    double error_cov_xx = Math::abs((double)base.cov_xx - (double)rotated_result.cov_xx);
    double error_cov_yy = Math::abs((double)base.cov_yy - (double)rotated_result.cov_yy);
    double error_cov_xy = Math::abs((double)base.cov_xy - (double)rotated_result.cov_xy);
    double error_conic_x = Math::abs((double)base.conic.x - (double)rotated_result.conic.x);
    double error_conic_y = Math::abs((double)base.conic.y - (double)rotated_result.conic.y);
    double error_conic_z = Math::abs((double)base.conic.z - (double)rotated_result.conic.z);

    return MAX(error_cov_xx, MAX(error_cov_yy, MAX(error_cov_xy, MAX(error_conic_x, MAX(error_conic_y, error_conic_z)))));
}

Error ProjectionValidator::generate_visualization(const String &p_path, const Vector<ProjectedGaussian> &p_cpu_results, const Vector<ProjectedGaussian> &p_gpu_results) const {
    ERR_FAIL_COND_V(p_cpu_results.size() != p_gpu_results.size(), ERR_INVALID_PARAMETER);

    if (p_cpu_results.is_empty()) {
        return ERR_INVALID_PARAMETER;
    }

    const int metric_count = 7; // cov_xx, cov_yy, cov_xy, determinant, conic.x, conic.y, conic.z
    int width = p_cpu_results.size();
    int height = metric_count;

    float max_error = 0.0f;
    Vector<Vector<float>> errors;
    ERR_FAIL_COND_V(errors.resize(metric_count) != OK, ERR_OUT_OF_MEMORY);
    for (int m = 0; m < metric_count; ++m) {
        ERR_FAIL_COND_V(errors.write[m].resize(width) != OK, ERR_OUT_OF_MEMORY);
    }

    for (int x = 0; x < width; ++x) {
        const ProjectedGaussian &cpu = p_cpu_results[x];
        const ProjectedGaussian &gpu = p_gpu_results[x];

        float metrics[metric_count] = {
            Math::abs(cpu.cov_xx - gpu.cov_xx),
            Math::abs(cpu.cov_yy - gpu.cov_yy),
            Math::abs(cpu.cov_xy - gpu.cov_xy),
            Math::abs(cpu.determinant - gpu.determinant),
            (float)Math::abs(cpu.conic.x - gpu.conic.x),
            (float)Math::abs(cpu.conic.y - gpu.conic.y),
            (float)Math::abs(cpu.conic.z - gpu.conic.z)
        };

        for (int m = 0; m < metric_count; ++m) {
            Vector<float> &row = errors.write[m];
            row.write[x] = metrics[m];
            max_error = MAX(max_error, metrics[m]);
        }
    }

    Ref<Image> image = Image::create_empty(width, height, false, Image::FORMAT_RGBA8);

    double denom = max_error > 0.0f ? max_error : 1.0f;
    for (int y = 0; y < height; ++y) {
        const Vector<float> &row = errors[y];
        for (int x = 0; x < width; ++x) {
            float normalized = row[x] / denom;
            normalized = CLAMP(normalized, 0.0f, 1.0f);
            Color color(normalized, 1.0f - normalized, 0.0f, 1.0f);
            image->set_pixel(x, y, color);
        }
    }

    String base_dir = p_path.get_base_dir();
    if (!base_dir.is_empty()) {
        DirAccess::make_dir_recursive_absolute(base_dir);
    }

    Error err = image->save_png(p_path);
    return err;
}

String ProjectionValidator::resolve_file_to_string(const String &p_path) const {
    Vector<String> attempts;
    attempts.push_back(p_path);
    if (!p_path.begins_with("res://")) {
        attempts.push_back(String("res://") + p_path);
    }

    for (int i = 0; i < attempts.size(); ++i) {
        const String &candidate = attempts[i];
        Ref<FileAccess> file = FileAccess::open(candidate, FileAccess::READ);
        if (file.is_valid()) {
            return file->get_as_text();
        }
    }

    return String();
}

Error ProjectionValidator::ensure_shader() {
    if (shader.is_valid() && pipeline.is_valid()) {
        return OK;
    }

    ERR_FAIL_COND_V(rd == nullptr, ERR_UNCONFIGURED);

    String include_source = resolve_file_to_string("modules/gaussian_splatting/shaders/includes/painterly_common.glsl");
    ERR_FAIL_COND_V(include_source.is_empty(), ERR_FILE_NOT_FOUND);

    String shader_source;
    shader_source += "#version 450\n";
    shader_source += include_source;
    shader_source += R"(
layout(local_size_x = 64) in;

struct ProjectionInput {
    vec4 view_col0;
    vec4 view_col1;
    vec4 view_col2;
    vec4 cov_col0;
    vec4 cov_col1;
    vec4 cov_col2;
};

struct ProjectionOutput {
    vec4 conic_det;
    vec4 covariance;
};

layout(push_constant) uniform Params {
    uint count;
} params;

layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {
    ProjectionInput inputs[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputBuffer {
    ProjectionOutput outputs[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= params.count) {
        return;
    }

    ProjectionInput input = inputs[index];
    mat3 view_matrix = mat3(input.view_col0.xyz, input.view_col1.xyz, input.view_col2.xyz);
    mat3 covariance_matrix = mat3(input.cov_col0.xyz, input.cov_col1.xyz, input.cov_col2.xyz);

    PainterlyConicData result = painterly_project_gaussian(view_matrix, covariance_matrix);
    outputs[index].conic_det = vec4(result.conic, result.determinant);
    outputs[index].covariance = vec4(result.cov_xx, result.cov_yy, result.cov_xy, 0.0);
}
)";

    Vector<uint8_t> spirv = rd->shader_compile_spirv_from_source(RD::SHADER_STAGE_COMPUTE, shader_source);
    ERR_FAIL_COND_V(spirv.is_empty(), ERR_PARSE_ERROR);

    Vector<RenderingDevice::ShaderStageSPIRVData> stages;
    stages.resize(1);
    stages.write[0].shader_stage = RD::SHADER_STAGE_COMPUTE;
    stages.write[0].spirv = spirv;

    shader = rd->shader_create_from_spirv(stages);
    ERR_FAIL_COND_V(!shader.is_valid(), ERR_CANT_CREATE);

    pipeline = rd->compute_pipeline_create(shader);
    ERR_FAIL_COND_V(!pipeline.is_valid(), ERR_CANT_CREATE);

    return OK;
}

Error ProjectionValidator::load_ground_truth(const String &p_path, Vector<ProjectionInput> &r_cases, String &r_error) const {
    String json_text = resolve_file_to_string(p_path);
    if (json_text.is_empty()) {
        r_error = "Failed to open ground truth JSON at " + p_path;
        return ERR_FILE_NOT_FOUND;
    }

    JSON json;
    Error err = json.parse(json_text);
    if (err != OK) {
        r_error = vformat("JSON parse error at line %d: %s", json.get_error_line(), json.get_error_message());
        return err;
    }

    Variant data = json.get_data();
    if (data.get_type() != Variant::DICTIONARY) {
        r_error = "Ground truth root must be a dictionary";
        return ERR_PARSE_ERROR;
    }

    Dictionary dict = data;
    if (!dict.has("cases")) {
        r_error = "Ground truth JSON missing 'cases' array";
        return ERR_PARSE_ERROR;
    }

    Array cases = dict["cases"];
    r_cases.clear();
    for (int i = 0; i < cases.size(); ++i) {
        Dictionary entry = cases[i];
        ProjectionInput input;
        input.name = entry.get("name", String("case_") + itos(i));

        Array scale_array = entry.get("scale", Array());
        ERR_FAIL_COND_V(scale_array.size() != 3, ERR_PARSE_ERROR);
        input.scale = Vector3((double)scale_array[0], (double)scale_array[1], (double)scale_array[2]);

        Array rotation_array = entry.get("rotation_quaternion", Array());
        ERR_FAIL_COND_V(rotation_array.size() != 4, ERR_PARSE_ERROR);
        input.rotation = Quaternion((double)rotation_array[0], (double)rotation_array[1], (double)rotation_array[2], (double)rotation_array[3]);

        Array view_rows = entry.get("view_rows", Array());
        ERR_FAIL_COND_V(view_rows.size() != 3, ERR_PARSE_ERROR);
        for (int r = 0; r < 3; ++r) {
            Array row = view_rows[r];
            ERR_FAIL_COND_V(row.size() != 3, ERR_PARSE_ERROR);
            input.view_basis[r] = Vector3((double)row[0], (double)row[1], (double)row[2]);
        }

        Dictionary expected = entry.get("expected", Dictionary());
        input.expected.cov_xx = (float)expected.get("cov_xx", 0.0);
        input.expected.cov_yy = (float)expected.get("cov_yy", 0.0);
        input.expected.cov_xy = (float)expected.get("cov_xy", 0.0);
        input.expected.determinant = (float)expected.get("determinant", 0.0);

        Array conic_array = expected.get("conic", Array());
        if (conic_array.size() == 3) {
            input.expected.conic = Vector3((double)conic_array[0], (double)conic_array[1], (double)conic_array[2]);
        } else {
            input.expected.conic = Vector3();
        }

        r_cases.push_back(input);
    }

    return OK;
}

} // namespace GaussianProjectionValidation
