#include "modules/gaussian_splatting/tests/test_macros.h"
#include "projection_validator.h"

#include "core/io/file_access.h"
#include "core/math/basis.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/math/random_number_generator.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"

using namespace GaussianProjectionValidation;

namespace {

constexpr uint64_t RANDOM_ORIENTATION_SEED = 0x4D595DF4ULL;
constexpr uint64_t PERFORMANCE_BENCHMARK_SEED = 0xC1A0C1A0ULL;

static bool approx_equal(float p_a, float p_b, float p_rel_tol = 1e-4f, float p_abs_tol = 1e-5f) {
    float diff = Math::abs(p_a - p_b);
    if (diff <= p_abs_tol) {
        return true;
    }
    float largest = MAX(Math::abs(p_a), Math::abs(p_b));
    if (largest == 0.0f) {
        return diff <= p_abs_tol;
    }
    return diff <= p_rel_tol * largest;
}

static Basis basis_from_euler_degrees(float p_pitch_deg, float p_yaw_deg, float p_roll_deg) {
    Vector3 euler_rad(Math::deg_to_rad(p_pitch_deg), Math::deg_to_rad(p_yaw_deg), Math::deg_to_rad(p_roll_deg));
    return Basis::from_euler(euler_rad, EulerOrder::XYZ);
}

static Quaternion random_unit_quaternion(RandomNumberGenerator &p_rng) {
    float u1 = p_rng.randf();
    float u2 = p_rng.randf() * (float)Math::TAU;
    float u3 = p_rng.randf() * (float)Math::TAU;
    float sqrt1 = Math::sqrt(1.0f - u1);
    float sqrt2 = Math::sqrt(u1);
    return Quaternion(Math::sin(u2) * sqrt1, Math::cos(u2) * sqrt1, Math::sin(u3) * sqrt2, Math::cos(u3) * sqrt2);
}

static void seed_rng(RandomNumberGenerator &p_rng, uint64_t p_seed) {
    p_rng.set_seed(p_seed);
}

} // namespace

TEST_CASE("[GaussianSplatting][Projection] Ground truth regression cases") {
    RenderingServer *rs = RenderingServer::get_singleton();
    REQUIRE(rs != nullptr);
    RenderingDevice *rd = rs->create_local_rendering_device();
    REQUIRE(rd != nullptr);

    ProjectionValidator validator;
    REQUIRE(validator.initialize(rd) == OK);

    Vector<ProjectionInput> ground_truth;
    String load_error;
    Error load_err = validator.load_ground_truth("tests/data/projection_ground_truth.json", ground_truth, load_error);
    INFO(vformat("Ground truth load error: %s", load_error));
    REQUIRE(load_err == OK);
    REQUIRE(!ground_truth.is_empty());

    Vector<ProjectedGaussian> gpu_results;
    REQUIRE(validator.project_gpu(ground_truth, gpu_results) == OK);
    REQUIRE(gpu_results.size() == ground_truth.size());

    Vector<ProjectedGaussian> cpu_results;
    REQUIRE(cpu_results.resize(ground_truth.size()) == OK);

    for (int i = 0; i < ground_truth.size(); ++i) {
        const ProjectionInput &test_case = ground_truth[i];
        ProjectedGaussian cpu = validator.project_cpu(test_case);
        cpu_results.write[i] = cpu;

        INFO(vformat("Case: %s", test_case.name));
        CHECK(approx_equal(cpu.cov_xx, test_case.expected.cov_xx));
        CHECK(approx_equal(cpu.cov_yy, test_case.expected.cov_yy));
        CHECK(approx_equal(cpu.cov_xy, test_case.expected.cov_xy, 1e-3f, 1e-4f));
        CHECK(approx_equal(cpu.determinant, test_case.expected.determinant, 1e-3f, 1e-3f));
        CHECK(approx_equal(cpu.conic.x, (float)test_case.expected.conic.x, 1e-4f, 1e-4f));
        CHECK(approx_equal(cpu.conic.y, (float)test_case.expected.conic.y, 1e-4f, 1e-4f));
        CHECK(approx_equal(cpu.conic.z, (float)test_case.expected.conic.z, 1e-4f, 1e-4f));

        const ProjectedGaussian &gpu = gpu_results[i];
        CHECK(approx_equal(gpu.cov_xx, cpu.cov_xx, 1e-4f, 1e-4f));
        CHECK(approx_equal(gpu.cov_yy, cpu.cov_yy, 1e-4f, 1e-4f));
        CHECK(approx_equal(gpu.cov_xy, cpu.cov_xy, 1e-3f, 1e-4f));
        CHECK(approx_equal(gpu.determinant, cpu.determinant, 1e-3f, 1e-3f));
        CHECK(approx_equal(gpu.conic.x, (float)cpu.conic.x, 1e-3f, 1e-4f));
        CHECK(approx_equal(gpu.conic.y, (float)cpu.conic.y, 1e-3f, 1e-4f));
        CHECK(approx_equal(gpu.conic.z, (float)cpu.conic.z, 1e-3f, 1e-4f));

        CHECK(cpu.cov_xx >= validator.get_min_variance());
        CHECK(cpu.cov_yy >= validator.get_min_variance());
    }

    Error viz_err = validator.generate_visualization("tests/output/projection_validation.png", cpu_results, gpu_results);
    CHECK(viz_err == OK);
    CHECK(FileAccess::exists("tests/output/projection_validation.png"));

    validator.shutdown();
    memdelete(rd);
}

TEST_CASE("[GaussianSplatting][Projection] Numerical stability edge cases") {
    RenderingServer *rs = RenderingServer::get_singleton();
    REQUIRE(rs != nullptr);
    RenderingDevice *rd = rs->create_local_rendering_device();
    REQUIRE(rd != nullptr);

    ProjectionValidator validator;
    REQUIRE(validator.initialize(rd) == OK);

    Vector<ProjectionInput> ground_truth;
    String load_error;
    REQUIRE(validator.load_ground_truth("tests/data/projection_ground_truth.json", ground_truth, load_error) == OK);

    int near_zero_index = -1;
    for (int i = 0; i < ground_truth.size(); ++i) {
        if (ground_truth[i].name.find("near_zero") != -1) {
            near_zero_index = i;
            break;
        }
    }
    REQUIRE(near_zero_index >= 0);

    const ProjectionInput &edge_case = ground_truth[near_zero_index];
    ProjectedGaussian cpu_result = validator.project_cpu(edge_case);

    Vector<ProjectionInput> single_case;
    REQUIRE(single_case.resize(1) == OK);
    single_case.write[0] = edge_case;

    Vector<ProjectedGaussian> gpu_result;
    REQUIRE(validator.project_gpu(single_case, gpu_result) == OK);
    REQUIRE(gpu_result.size() == 1);

    const ProjectedGaussian &gpu = gpu_result[0];

    CHECK(approx_equal(cpu_result.determinant, 0.0f, 1e-3f, 1e-3f));
    CHECK(approx_equal(gpu.determinant, cpu_result.determinant, 1e-3f, 1e-3f));
    CHECK(Math::is_finite((float)cpu_result.conic.x));
    CHECK(Math::is_finite((float)cpu_result.conic.y));
    CHECK(Math::is_finite((float)cpu_result.conic.z));
    CHECK(Math::is_finite((float)gpu.conic.x));
    CHECK(Math::is_finite((float)gpu.conic.y));
    CHECK(Math::is_finite((float)gpu.conic.z));

    CHECK(cpu_result.cov_xx >= validator.get_min_variance());
    CHECK(cpu_result.cov_yy >= validator.get_min_variance());

    validator.shutdown();
    memdelete(rd);
}

TEST_CASE("[GaussianSplatting][Projection] Random orientation statistical validation") {
    RenderingServer *rs = RenderingServer::get_singleton();
    REQUIRE(rs != nullptr);
    RenderingDevice *rd = rs->create_local_rendering_device();
    REQUIRE(rd != nullptr);

    ProjectionValidator validator;
    REQUIRE(validator.initialize(rd) == OK);

    RandomNumberGenerator rng;
    seed_rng(rng, RANDOM_ORIENTATION_SEED);
    INFO(vformat("RNG seed: %s", itos((int64_t)RANDOM_ORIENTATION_SEED)));

    const int sample_count = 64;
    Vector<ProjectionInput> samples;
    REQUIRE(samples.resize(sample_count) == OK);

    for (int i = 0; i < sample_count; ++i) {
        ProjectionInput sample;
        sample.name = String("random_") + itos(i);
        sample.scale = Vector3(rng.randf_range(0.2f, 4.0f), rng.randf_range(0.2f, 4.0f), rng.randf_range(0.2f, 4.0f));
        sample.rotation = random_unit_quaternion(rng);
        Vector3 view_euler(rng.randf_range((float)-Math::PI, (float)Math::PI), rng.randf_range((float)-Math::PI, (float)Math::PI), rng.randf_range((float)-Math::PI, (float)Math::PI));
        sample.view_basis = Basis::from_euler(view_euler, EulerOrder::XYZ);
        samples.write[i] = sample;
    }

    Vector<ProjectedGaussian> gpu_results;
    REQUIRE(validator.project_gpu(samples, gpu_results) == OK);

    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    int measurement_count = 0;

    for (int i = 0; i < sample_count; ++i) {
        ProjectedGaussian cpu = validator.project_cpu(samples[i]);
        const ProjectedGaussian &gpu = gpu_results[i];

        float errors[6] = {
            Math::abs(cpu.cov_xx - gpu.cov_xx),
            Math::abs(cpu.cov_yy - gpu.cov_yy),
            Math::abs(cpu.cov_xy - gpu.cov_xy),
            Math::abs(cpu.determinant - gpu.determinant),
            (float)Math::abs(cpu.conic.x - gpu.conic.x),
            (float)Math::abs(cpu.conic.y - gpu.conic.y)
        };
        float conic_error_z = (float)Math::abs(cpu.conic.z - gpu.conic.z);

        for (float error_value : errors) {
            max_abs_error = MAX(max_abs_error, (double)error_value);
            mean_abs_error += error_value;
            ++measurement_count;
        }
        max_abs_error = MAX(max_abs_error, (double)conic_error_z);
        mean_abs_error += conic_error_z;
        ++measurement_count;

        double lambda_min = 0.0;
        double lambda_max = 0.0;
        validator.compute_eigenvalues(cpu, lambda_min, lambda_max);
        CHECK(lambda_min >= -1e-5);
        validator.compute_eigenvalues(gpu, lambda_min, lambda_max);
        CHECK(lambda_min >= -1e-5);
    }

    mean_abs_error /= MAX(measurement_count, 1);
    INFO(vformat("Max absolute difference: %.6f", max_abs_error));
    INFO(vformat("Mean absolute difference: %.6f", mean_abs_error));

    CHECK(max_abs_error < 1e-3);
    CHECK(mean_abs_error < 1e-4);

    validator.shutdown();
    memdelete(rd);
}

TEST_CASE("[GaussianSplatting][Projection] Camera angle coverage") {
    RenderingServer *rs = RenderingServer::get_singleton();
    REQUIRE(rs != nullptr);
    RenderingDevice *rd = rs->create_local_rendering_device();
    REQUIRE(rd != nullptr);

    ProjectionValidator validator;
    REQUIRE(validator.initialize(rd) == OK);

    const Vector<Vector3> camera_angles = {
        Vector3(0.0, 0.0, 0.0),
        Vector3(15.0, 30.0, 0.0),
        Vector3(-25.0, 10.0, 5.0),
        Vector3(45.0, -60.0, 10.0)
    };

    Vector<ProjectionInput> inputs;
    REQUIRE(inputs.resize(camera_angles.size()) == OK);

    for (int i = 0; i < camera_angles.size(); ++i) {
        ProjectionInput input;
        input.name = String("camera_") + itos(i);
        input.scale = Vector3(1.2f, 0.8f, 2.5f);
        input.rotation = Quaternion();
        const Vector3 &angles = camera_angles[i];
        input.view_basis = basis_from_euler_degrees(angles.x, angles.y, angles.z);
        inputs.write[i] = input;
    }

    Vector<ProjectedGaussian> gpu_results;
    REQUIRE(validator.project_gpu(inputs, gpu_results) == OK);

    for (int i = 0; i < inputs.size(); ++i) {
        ProjectedGaussian cpu = validator.project_cpu(inputs[i]);
        const ProjectedGaussian &gpu = gpu_results[i];

        CHECK(approx_equal(cpu.cov_xx, gpu.cov_xx, 1e-4f, 1e-4f));
        CHECK(approx_equal(cpu.cov_yy, gpu.cov_yy, 1e-4f, 1e-4f));
        CHECK(approx_equal(cpu.cov_xy, gpu.cov_xy, 1e-4f, 1e-4f));
        CHECK(approx_equal(cpu.determinant, gpu.determinant, 1e-4f, 1e-4f));

        double lambda_min = 0.0;
        double lambda_max = 0.0;
        validator.compute_eigenvalues(cpu, lambda_min, lambda_max);
        CHECK(lambda_min >= -1e-5);
    }

    validator.shutdown();
    memdelete(rd);
}

TEST_CASE("[GaussianSplatting][Projection] Scale and rotation invariants") {
    RenderingServer *rs = RenderingServer::get_singleton();
    REQUIRE(rs != nullptr);
    RenderingDevice *rd = rs->create_local_rendering_device();
    REQUIRE(rd != nullptr);

    ProjectionValidator validator;
    REQUIRE(validator.initialize(rd) == OK);

    ProjectionInput base_case;
    base_case.name = "invariance";
    base_case.scale = Vector3(0.75f, 1.5f, 2.25f);
    base_case.rotation = Quaternion(Vector3(0.3f, 0.7f, -0.2f).normalized(), Math::deg_to_rad(37.0f));
    base_case.view_basis = Basis::from_euler(Vector3(Math::deg_to_rad(25.0f), Math::deg_to_rad(-15.0f), Math::deg_to_rad(5.0f)), EulerOrder::XYZ);

    double scale_error = validator.compute_scale_invariance_error(base_case, 1.8f);
    CHECK(scale_error < 5e-3);

    Basis additional_rotation = Basis::from_euler(Vector3(Math::deg_to_rad(12.0f), Math::deg_to_rad(-9.0f), Math::deg_to_rad(3.0f)), EulerOrder::XYZ);
    double rotation_error = validator.compute_rotation_equivariance_error(base_case, additional_rotation);
    CHECK(rotation_error < 5e-3);

    validator.shutdown();
    memdelete(rd);
}

TEST_CASE("[GaussianSplatting][Projection] CPU vs GPU performance benchmark") {
    RenderingServer *rs = RenderingServer::get_singleton();
    REQUIRE(rs != nullptr);
    RenderingDevice *rd = rs->create_local_rendering_device();
    REQUIRE(rd != nullptr);

    ProjectionValidator validator;
    REQUIRE(validator.initialize(rd) == OK);

    const int benchmark_count = 4096;
    Vector<ProjectionInput> inputs;
    REQUIRE(inputs.resize(benchmark_count) == OK);

    RandomNumberGenerator rng;
    seed_rng(rng, PERFORMANCE_BENCHMARK_SEED);
    INFO(vformat("RNG seed: %s", itos((int64_t)PERFORMANCE_BENCHMARK_SEED)));

    for (int i = 0; i < benchmark_count; ++i) {
        ProjectionInput input;
        input.scale = Vector3(rng.randf_range(0.5f, 3.0f), rng.randf_range(0.5f, 3.0f), rng.randf_range(0.5f, 3.0f));
        input.rotation = random_unit_quaternion(rng);
        input.view_basis = Basis::from_euler(Vector3(rng.randf_range((float)-Math::PI, (float)Math::PI), rng.randf_range((float)-Math::PI, (float)Math::PI), rng.randf_range((float)-Math::PI, (float)Math::PI)), EulerOrder::XYZ);
        inputs.write[i] = input;
    }

    uint64_t cpu_start = OS::get_singleton()->get_ticks_usec();
    Vector<ProjectedGaussian> cpu_results;
    REQUIRE(cpu_results.resize(benchmark_count) == OK);
    for (int i = 0; i < benchmark_count; ++i) {
        cpu_results.write[i] = validator.project_cpu(inputs[i]);
    }
    uint64_t cpu_time = OS::get_singleton()->get_ticks_usec() - cpu_start;

    Vector<ProjectedGaussian> gpu_results;
    uint64_t gpu_start = OS::get_singleton()->get_ticks_usec();
    REQUIRE(validator.project_gpu(inputs, gpu_results) == OK);
    uint64_t gpu_time = OS::get_singleton()->get_ticks_usec() - gpu_start;

    double cpu_ms = cpu_time / 1000.0;
    double gpu_ms = gpu_time / 1000.0;

    INFO(vformat("CPU projection time: %.3f ms", cpu_ms));
    INFO(vformat("GPU projection time: %.3f ms", gpu_ms));

    CHECK_MESSAGE(gpu_ms <= cpu_ms * 2.5 + 0.5, "GPU projection should not be dramatically slower than CPU.");

    validator.shutdown();
    memdelete(rd);
}
