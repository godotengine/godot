/*
 * Visual Validation System for Gaussian Splatting
 * Ensures rendering correctness through image comparison and perceptual metrics.
 */

#ifndef VISUAL_VALIDATION_H
#define VISUAL_VALIDATION_H

#include "core/object/ref_counted.h"
#include "core/io/image.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"

class RenderingDevice;
class Viewport;

// Image comparison metrics
struct ImageMetrics {
    float ssim = 0.0f;        // Structural Similarity Index
    float psnr = 0.0f;        // Peak Signal-to-Noise Ratio (dB)
    float mae = 0.0f;         // Mean Absolute Error
    float mse = 0.0f;         // Mean Squared Error
    float max_error = 0.0f;   // Maximum pixel difference
    float perceptual = 0.0f;  // Perceptual difference metric
    uint32_t different_pixels = 0;  // Count of different pixels
    float difference_ratio = 0.0f;  // Percentage of different pixels

    bool passes_threshold(float ssim_threshold = 0.95f) const {
        return ssim >= ssim_threshold;
    }

    String to_string() const;
    Dictionary to_dict() const;
};

// Visual test configuration
struct VisualTestConfig {
    String name;
    uint32_t viewport_width = 1920;
    uint32_t viewport_height = 1080;
    uint32_t splat_count = 100000;
    String data_pattern = "uniform";  // uniform, clustered, etc.
    Vector3 camera_position = Vector3(0, 0, 10);
    Vector3 camera_target = Vector3(0, 0, 0);
    float fov = 60.0f;
    bool enable_sorting = true;
    bool enable_culling = true;
    float ssim_threshold = 0.95f;
    float psnr_threshold = 30.0f;  // dB

    Dictionary to_dict() const;
    void from_dict(const Dictionary &dict);
};

// Visual test result
struct VisualTestResult {
    String test_name;
    bool passed = false;
    ImageMetrics metrics;
    String reference_path;
    String result_path;
    String diff_path;
    String error_message;
    float render_time_ms = 0.0f;

    Dictionary to_dict() const;
    String generate_report() const;
};

// Main visual validation class
class VisualValidation : public RefCounted {
    GDCLASS(VisualValidation, RefCounted);

protected:
    static void _bind_methods();

private:
    RenderingDevice *rd = nullptr;
    Viewport *test_viewport = nullptr;

    // Reference image storage
    HashMap<String, Ref<Image>> reference_images;
    String reference_dir = "res://tests/visual_references/";
    String output_dir = "res://tests/visual_output/";

    // Test configurations
    LocalVector<VisualTestConfig> test_configs;

    // Helper methods
    Ref<Image> render_frame(const VisualTestConfig &config);
    Ref<Image> load_reference(const String &name);
    void save_reference(const String &name, const Ref<Image> &image);
    Ref<Image> generate_diff_image(const Ref<Image> &ref, const Ref<Image> &result);

public:
    VisualValidation();
    ~VisualValidation();

    // Initialize with rendering device
    Error initialize(RenderingDevice *p_rd);

    // Reference management
    void capture_reference_frames();
    void capture_reference(const String &name, const VisualTestConfig &config);
    bool has_reference(const String &name) const;
    void clear_references();
    Error load_references_from_disk();
    Error save_references_to_disk();

    // Image comparison methods
    static ImageMetrics compare_images(const Ref<Image> &ref, const Ref<Image> &result);
    static float calculate_ssim(const Ref<Image> &a, const Ref<Image> &b);
    static float calculate_psnr(const Ref<Image> &a, const Ref<Image> &b);
    static float calculate_mae(const Ref<Image> &a, const Ref<Image> &b);
    static float calculate_perceptual_diff(const Ref<Image> &a, const Ref<Image> &b);

    // Visual tests
    VisualTestResult run_visual_test(const VisualTestConfig &config);
    LocalVector<VisualTestResult> run_all_visual_tests();
    VisualTestResult validate_rendering(const String &test_name);
    VisualTestResult validate_depth_ordering();
    VisualTestResult validate_transparency();
    VisualTestResult validate_culling();
    VisualTestResult validate_lod_transitions();

    // Temporal stability tests
    VisualTestResult validate_temporal_stability(uint32_t frame_count = 10);
    float detect_flickering(const LocalVector<Ref<Image>> &frames);
    float detect_popping(const LocalVector<Ref<Image>> &frames);

    // Platform consistency tests
    VisualTestResult validate_platform_consistency();
    ImageMetrics compare_with_platform(const String &platform_name);

    // Quality tests
    VisualTestResult validate_antialiasing();
    VisualTestResult validate_color_accuracy();
    VisualTestResult validate_edge_quality();

    // Reporting
    void generate_visual_report(const String &output_file);
    void generate_html_gallery(const String &output_file);
    void export_test_results(const LocalVector<VisualTestResult> &results, const String &filepath);

    // Configuration
    void set_reference_dir(const String &dir) { reference_dir = dir; }
    void set_output_dir(const String &dir) { output_dir = dir; }
    void add_test_config(const VisualTestConfig &config) { test_configs.push_back(config); }
    void load_test_configs(const String &config_file);
};

// SSIM calculation helper
class SSIMCalculator {
private:
    static constexpr float K1 = 0.01f;
    static constexpr float K2 = 0.03f;
    static constexpr float L = 255.0f;  // Dynamic range
    static constexpr int WINDOW_SIZE = 11;

    static float gaussian_weight(int x, int y, float sigma);
    static void apply_gaussian_window(const Ref<Image> &img, int x, int y,
                                     float &mean, float &variance, float &std_dev);

public:
    static float compute(const Ref<Image> &img1, const Ref<Image> &img2);
    static Ref<Image> generate_ssim_map(const Ref<Image> &img1, const Ref<Image> &img2);
};

// PSNR calculation helper
class PSNRCalculator {
public:
    static float compute(const Ref<Image> &img1, const Ref<Image> &img2);
    static float compute_mse(const Ref<Image> &img1, const Ref<Image> &img2);
};

// Perceptual difference calculator
class PerceptualDiff {
private:
    // LAB color space conversion
    static Vector3 rgb_to_lab(const Color &rgb);
    static float lab_distance(const Vector3 &lab1, const Vector3 &lab2);

public:
    static float compute(const Ref<Image> &img1, const Ref<Image> &img2);
    static Ref<Image> generate_perceptual_diff_map(const Ref<Image> &img1, const Ref<Image> &img2);
};

// Visual test suite
class VisualTestSuite {
private:
    Ref<VisualValidation> validator;
    LocalVector<VisualTestResult> results;

public:
    VisualTestSuite();

    // Standard test scenarios
    void test_basic_rendering();
    void test_large_dataset();
    void test_transparency_sorting();
    void test_view_angles();
    void test_zoom_levels();
    void test_lighting_conditions();
    void test_motion_blur();

    // Regression tests
    void test_against_baseline();
    void test_version_compatibility();

    // Performance visual tests
    void test_lod_quality();
    void test_culling_correctness();
    void test_streaming_quality();

    // Run all tests
    void run_all();
    bool check_all_passed() const;
    void generate_report(const String &filepath);
};

// Golden image manager
class GoldenImageManager {
private:
    String golden_dir = "res://tests/golden_images/";
    HashMap<String, Ref<Image>> golden_images;
    HashMap<String, VisualTestConfig> golden_configs;

public:
    // Golden image management
    Error capture_golden(const String &name, const VisualTestConfig &config);
    Ref<Image> get_golden(const String &name);
    bool has_golden(const String &name) const;

    // Version management
    void tag_golden_set(const String &version);
    void load_golden_set(const String &version);

    // Comparison with golden
    ImageMetrics compare_with_golden(const String &name, const Ref<Image> &image);
    LocalVector<String> list_golden_images() const;

    // Export/Import
    Error export_golden_set(const String &filepath);
    Error import_golden_set(const String &filepath);
};

// Visual debugging tools
class VisualDebugger {
public:
    // Generate debug visualizations
    static Ref<Image> visualize_depth_buffer(const Ref<Image> &depth);
    static Ref<Image> visualize_sorting_order(const LocalVector<uint32_t> &indices);
    static Ref<Image> visualize_culling_mask(const LocalVector<bool> &mask);
    static Ref<Image> visualize_lod_levels(const LocalVector<uint8_t> &levels);

    // Diff visualizations
    static Ref<Image> create_diff_heatmap(const Ref<Image> &ref, const Ref<Image> &result);
    static Ref<Image> create_side_by_side(const Ref<Image> &ref, const Ref<Image> &result);
    static Ref<Image> create_overlay(const Ref<Image> &ref, const Ref<Image> &result, float alpha = 0.5f);
    static Ref<Image> create_checkerboard(const Ref<Image> &ref, const Ref<Image> &result, int size = 32);

    // Analysis visualizations
    static Ref<Image> highlight_differences(const Ref<Image> &ref, const Ref<Image> &result, float threshold = 0.1f);
    static Ref<Image> visualize_ssim_map(const Ref<Image> &ssim_map);
    static Ref<Image> visualize_error_distribution(const Ref<Image> &ref, const Ref<Image> &result);
};

#endif // VISUAL_VALIDATION_H