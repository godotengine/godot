#ifndef PLY_LOADER_H
#define PLY_LOADER_H

#include "core/io/file_access.h"
#include "core/object/ref_counted.h"
#include "../core/gaussian_data.h"
#include "core/variant/variant.h"
#include "core/variant/dictionary.h"
#include <cstdint>

/**
 * @class PLYLoader
 * @brief Loader for PLY (Polygon File Format) Gaussian Splatting data.
 *
 * PLY is the standard interchange format for Gaussian Splatting. This loader
 * supports both ASCII and binary (little/big endian) PLY files containing
 * Gaussian splat vertex data.
 *
 * ## Supported Properties
 *
 * **Required properties** (loader fails without these):
 * - x, y, z: Position coordinates (float)
 * - f_dc_0, f_dc_1, f_dc_2: Base color as spherical harmonic DC term (float)
 * - scale_0, scale_1, scale_2: Log-encoded scale factors (float)
 * - rot_0, rot_1, rot_2, rot_3: Rotation quaternion (float, WXYZ order)
 * - opacity: Logit-encoded opacity (float)
 *
 * **Optional properties** (loaded if present):
 * - nx, ny, nz: Surface normals for 2D Gaussian (surfel) mode
 * - f_rest_*: Higher-order spherical harmonic coefficients
 *
 * ## Format Notes
 *
 * - Scales are stored as **log scale** values: actual_scale = exp(stored_value)
 * - Opacity is stored as **logit** values: actual_opacity = sigmoid(stored_value)
 * - Spherical harmonics use the f_dc_* / f_rest_* naming convention from the
 *   original 3D Gaussian Splatting paper.
 *
 * ## Validation
 *
 * Use get_property_deficiencies() to check for missing required or optional
 * properties before or after loading. The has_property() and get_property_names()
 * methods allow inspection of the PLY header.
 *
 * @see SPZLoader for the compressed Niantic format alternative.
 */
class PLYLoader : public RefCounted {
    GDCLASS(PLYLoader, RefCounted);

public:
    struct PLYProperty {
        String name;
        String type;
        int offset = 0;
        int size = 0;
    };

    struct PLYHeader {
        bool is_binary = false;
        bool is_little_endian = true;
        int vertex_count = 0;
        Vector<PLYProperty> properties;
        int header_size = 0;
    };

private:
    PLYHeader header;
    Ref<::GaussianData> gaussian_data;
    uint64_t last_load_time_us = 0;
    uint64_t last_header_time_us = 0;
    uint64_t last_parse_time_us = 0;
    uint64_t last_cache_time_us = 0;
    bool last_cache_hit = false;

    Error parse_header(Ref<FileAccess> file);
    Error parse_ascii_data(Ref<FileAccess> file);
    Error parse_binary_data(Ref<FileAccess> file);
    bool try_load_cache(const String &p_source_path, uint64_t p_source_size, uint64_t p_source_mtime);
    void write_cache(const String &p_source_path, uint64_t p_source_size, uint64_t p_source_mtime) const;

    // Map PLY properties to Gaussian attributes
    int find_property_index(const String &name) const;
    float read_float_property(const uint8_t *data, const PLYProperty &prop) const;
    uint32_t read_uint_property(const uint8_t *data, const PLYProperty &prop) const;
    int assemble_sh_coefficients(Gaussian &r_gaussian,
            const float *p_dc_values,
            const bool *p_dc_present,
            const float *p_rest_values,
            const bool *p_rest_present,
            float *r_output) const;

protected:
    static void _bind_methods();

public:
    PLYLoader();
    ~PLYLoader();

    Error load_file(const String &p_path);
    Ref<::GaussianData> get_gaussian_data() const { return gaussian_data; }

    // Statistics
    int get_splat_count() const;
    Dictionary get_load_statistics() const;

    // Property analysis helpers
    void get_property_deficiencies(PackedStringArray &r_missing_required, PackedStringArray &r_missing_optional) const;
    Dictionary get_property_summary() const;

    // Validation support
    bool has_property(const String &p_name) const;
    Vector<String> get_property_names() const;
    PLYHeader get_header() const { return header; }
};

#endif // PLY_LOADER_H
