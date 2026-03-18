#ifndef SPZ_LOADER_H
#define SPZ_LOADER_H

#include "core/io/file_access.h"
#include "core/io/compression.h"
#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "core/variant/dictionary.h"
#include "../core/gaussian_data.h"

/**
 * @class SPZLoader
 * @brief Loader for Niantic's SPZ compressed Gaussian Splatting format.
 *
 * SPZ is a compressed format for Gaussian Splatting data that provides
 * approximately 10x compression over PLY with minimal visual degradation.
 * It is used by Scaniverse and other mobile applications.
 *
 * ## Format Specification
 *
 * Based on github.com/nianticlabs/spz:
 * - 16-byte header with magic number 0x5053474E ("NGSP")
 * - gzip-compressed data stream containing packed splat attributes
 *
 * ## Data Encoding
 *
 * **Positions**: 24-bit fixed-point signed integers (3 per point)
 * - Precision controlled by header.fractional_bits (typically 11-13 bits)
 * - Decoded as: position = fixed_value / (1 << fractional_bits)
 *
 * **Scales**: 8-bit log-encoded values (3 per point)
 * - Decoded as: scale = exp((encoded - 128) / 16.0)
 * - Range approximately [1e-5, 2e+4]
 *
 * **Rotations**: Smallest-three quaternion encoding
 * - Version 2: 3 bytes per rotation (8 bits per component)
 * - Version 3: 4 bytes per rotation (improved precision)
 * - Uses smallest-three encoding: store 3 smallest components, reconstruct 4th
 *
 * **Alphas**: 8-bit unsigned integers
 * - Decoded using inverse sigmoid: alpha = sigmoid((encoded - 128) / 32.0)
 *
 * **Colors**: RGB as 8-bit unsigned integers
 * - Direct byte values normalized to [0, 1]
 *
 * **Spherical Harmonics**: Variable coefficients based on SH degree (0-3)
 * - Stored as quantized values when sh_degree > 0
 *
 * ## Comparison with PLY
 *
 * | Feature | PLY | SPZ |
 * |---------|-----|-----|
 * | File size | Large (32-bit floats) | ~10x smaller |
 * | Position precision | Full float32 | Fixed-point (configurable) |
 * | Rotation precision | Full float32 | Smallest-three encoded |
 * | Validation | Property-level | Header-level |
 * | Editing | Human-readable (ASCII) | Binary only |
 *
 * @see PLYLoader for the uncompressed PLY format alternative.
 */
class SPZLoader : public RefCounted {
    GDCLASS(SPZLoader, RefCounted);

public:
    /**
     * @struct SPZHeader
     * @brief SPZ file header structure (16 bytes).
     */
    struct SPZHeader {
        uint32_t magic;           ///< Magic number: 0x5053474E ("NGSP" in little-endian)
        uint32_t version;         ///< Format version (2 or 3 supported)
        uint32_t num_points;      ///< Number of Gaussian points
        uint8_t sh_degree;        ///< Spherical harmonics degree (0-3)
        uint8_t fractional_bits;  ///< Fixed-point precision for positions
        uint8_t flags;            ///< Feature flags (bit 0x1 = antialiased training)
        uint8_t reserved;         ///< Must be zero
    };

    static constexpr uint32_t SPZ_MAGIC = 0x5053474E;  // "NGSP" in little-endian
    static constexpr uint32_t SPZ_VERSION_2 = 2;
    static constexpr uint32_t SPZ_VERSION_3 = 3;

    // Flag bit definitions
    static constexpr uint8_t SPZ_FLAG_ANTIALIASED = 0x01;

private:
    SPZHeader header;
    Ref<::GaussianData> gaussian_data;

    // Decompression helpers
    Error decompress_data(const PackedByteArray &p_compressed, PackedByteArray &r_decompressed,
            uint64_t p_max_decompressed_bytes, uint64_t p_expected_decompressed_bytes = 0);

    // Data parsing helpers
    Error parse_positions(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Vector3> &r_positions);
    Error parse_alphas(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<float> &r_alphas);
    Error parse_colors(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Color> &r_colors);
    Error parse_scales(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Vector3> &r_scales);
    Error parse_rotations_v2(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Quaternion> &r_rotations);
    Error parse_rotations_v3(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Quaternion> &r_rotations);
    Error parse_spherical_harmonics(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset,
            LocalVector<float> &r_sh_coeffs, uint32_t &r_sh_float_count_per_gaussian);

    // Fixed-point conversion
    float fixed_to_float(int32_t p_fixed, uint8_t p_fractional_bits) const;
    int32_t read_int24(const uint8_t *p_data) const;

    // Quantization helpers
    float decode_scale(uint8_t p_encoded) const;
    float decode_alpha(uint8_t p_encoded) const;
    float decode_sh_coefficient(uint8_t p_encoded, uint8_t p_bits) const;

protected:
    static void _bind_methods();

public:
    SPZLoader();
    ~SPZLoader();

    /**
     * @brief Loads an SPZ file from disk.
     * @param p_path Path to the SPZ file.
     * @return OK on success, or an error code.
     */
    Error load_file(const String &p_path);

    /**
     * @brief Returns the loaded Gaussian data.
     * @return Reference to the GaussianData resource.
     */
    Ref<::GaussianData> get_gaussian_data() const { return gaussian_data; }

    /**
     * @brief Returns the number of splats loaded.
     * @return Number of Gaussian splats.
     */
    int get_splat_count() const;

    /**
     * @brief Returns loading statistics.
     * @return Dictionary containing format info, splat count, and bounds.
     */
    Dictionary get_load_statistics() const;

    /**
     * @brief Returns the SPZ header information.
     * @return Copy of the parsed header.
     */
    SPZHeader get_header() const { return header; }

    /**
     * @brief Checks if the file at the given path is a valid SPZ file.
     * @param p_path Path to check.
     * @return true if the file has a valid SPZ magic number.
     */
    static bool is_spz_file(const String &p_path);
};

#endif // SPZ_LOADER_H
