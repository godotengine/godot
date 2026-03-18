#ifndef TESTS_PROJECTION_VALIDATOR_H
#define TESTS_PROJECTION_VALIDATOR_H

#include "core/io/file_access.h"
#include "core/math/basis.h"
#include "core/math/quaternion.h"
#include "core/math/vector3.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"

namespace GaussianProjectionValidation {

struct ProjectedGaussian {
    float cov_xx = 0.0f;
    float cov_yy = 0.0f;
    float cov_xy = 0.0f;
    float determinant = 0.0f;
    Vector3 conic;
};

struct ProjectionInput {
    String name;
    Vector3 scale;
    Quaternion rotation;
    Basis view_basis;
    ProjectedGaussian expected;
};

class ProjectionValidator {
public:
    ProjectionValidator();
    ~ProjectionValidator();

    Error initialize(RenderingDevice *p_rd);
    void shutdown();

    bool is_ready() const { return shader.is_valid() && pipeline.is_valid(); }

    Error load_ground_truth(const String &p_path, Vector<ProjectionInput> &r_cases, String &r_error) const;

    ProjectedGaussian project_cpu(const ProjectionInput &p_input) const;
    ProjectedGaussian project_cpu(const Basis &p_view_basis, const Quaternion &p_rotation, const Vector3 &p_scale) const;

    Error project_gpu(const Vector<ProjectionInput> &p_inputs, Vector<ProjectedGaussian> &r_results);

    void compute_eigenvalues(const ProjectedGaussian &p_result, double &r_lambda_min, double &r_lambda_max) const;
    double compute_positive_semidefinite_margin(const ProjectedGaussian &p_result) const;
    double compute_scale_invariance_error(const ProjectionInput &p_input, float p_uniform_scale) const;
    double compute_rotation_equivariance_error(const ProjectionInput &p_input, const Basis &p_rotation) const;

    Error generate_visualization(const String &p_path, const Vector<ProjectedGaussian> &p_cpu_results, const Vector<ProjectedGaussian> &p_gpu_results) const;

    float get_variance_epsilon() const { return variance_epsilon; }
    float get_min_variance() const { return min_variance; }
    float get_min_determinant() const { return min_determinant; }

private:
    RenderingDevice *rd = nullptr;
    RID shader;
    RID pipeline;

    static constexpr float variance_epsilon = 0.3f;
    static constexpr float min_variance = 0.0001f;
    static constexpr float min_determinant = 1e-8f;

    struct Matrix3f;

    Matrix3f basis_to_matrix(const Basis &p_basis) const;
    Matrix3f build_rotation_matrix(const Quaternion &p_rotation) const;
    Matrix3f build_covariance_matrix(const Quaternion &p_rotation, const Vector3 &p_scale) const;
    Matrix3f multiply(const Matrix3f &p_a, const Matrix3f &p_b) const;
    Matrix3f transpose(const Matrix3f &p_m) const;

    ProjectedGaussian project_from_matrices(const Matrix3f &p_view, const Matrix3f &p_covariance) const;

    String resolve_file_to_string(const String &p_path) const;
    Error ensure_shader();
};

} // namespace GaussianProjectionValidation

#endif // TESTS_PROJECTION_VALIDATOR_H
