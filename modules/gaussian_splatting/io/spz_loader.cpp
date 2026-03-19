#include "spz_loader.h"
#include "core/os/os.h"
#include "../logger/gs_logger.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace {

// SH coefficient counts per degree
static constexpr int SH_COEFFS_PER_DEGREE[] = { 0, 9, 24, 45 };

// SH bits per band
static constexpr uint8_t SH_BITS_DEGREE_0 = 5;
static constexpr uint8_t SH_BITS_DEGREE_1_2 = 4;

// Conversion constants
static constexpr float LOG_SCALE_MIN = -10.0f;
static constexpr float LOG_SCALE_MAX = 6.0f;
static constexpr uint64_t MAX_SPZ_COMPRESSED_BYTES = 512ull * 1024ull * 1024ull; // 512 MiB
static constexpr uint64_t MAX_SPZ_DECOMPRESSED_BYTES = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr uint32_t MAX_SPZ_POINTS = 32u * 1024u * 1024u; // 33,554,432 points

static bool _offset_range_valid(uint32_t p_offset, uint64_t p_needed, uint32_t p_total_size) {
    return uint64_t(p_offset) + p_needed <= uint64_t(p_total_size);
}

static bool _mul_u64_overflow(uint64_t p_a, uint64_t p_b, uint64_t &r_result) {
    if (p_a != 0 && p_b > (std::numeric_limits<uint64_t>::max() / p_a)) {
        return true;
    }
    r_result = p_a * p_b;
    return false;
}

static bool _add_u64_overflow(uint64_t p_a, uint64_t p_b, uint64_t &r_result) {
    if (p_b > (std::numeric_limits<uint64_t>::max() - p_a)) {
        return true;
    }
    r_result = p_a + p_b;
    return false;
}

static bool _compute_expected_payload_bytes(const SPZLoader::SPZHeader &p_header, uint64_t &r_expected_payload_bytes) {
    if (p_header.version != SPZLoader::SPZ_VERSION_2 && p_header.version != SPZLoader::SPZ_VERSION_3) {
        return false;
    }
    if (p_header.sh_degree > 3) {
        return false;
    }

    const uint64_t rotation_stride = (p_header.version == SPZLoader::SPZ_VERSION_2) ? 3ull : 4ull;
    uint64_t per_point_bytes = 9ull + 1ull + 3ull + 3ull + rotation_stride;
    if (p_header.sh_degree > 0) {
        uint64_t with_sh = 0;
        if (_add_u64_overflow(per_point_bytes, uint64_t(SH_COEFFS_PER_DEGREE[p_header.sh_degree]), with_sh)) {
            return false;
        }
        per_point_bytes = with_sh;
    }

    uint64_t payload_bytes = 0;
    if (_mul_u64_overflow(uint64_t(p_header.num_points), per_point_bytes, payload_bytes)) {
        return false;
    }

    r_expected_payload_bytes = payload_bytes;
    return true;
}

} // namespace

SPZLoader::SPZLoader() {
    gaussian_data.instantiate();
    header = SPZHeader{};
}

SPZLoader::~SPZLoader() {
}

void SPZLoader::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_file", "path"), &SPZLoader::load_file);
    ClassDB::bind_method(D_METHOD("get_gaussian_data"), &SPZLoader::get_gaussian_data);
    ClassDB::bind_method(D_METHOD("get_splat_count"), &SPZLoader::get_splat_count);
    ClassDB::bind_method(D_METHOD("get_load_statistics"), &SPZLoader::get_load_statistics);
}

bool SPZLoader::is_spz_file(const String &p_path) {
    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
    if (file.is_null()) {
        return false;
    }

    if (file->get_length() < sizeof(SPZHeader)) {
        return false;
    }

    // Check first bytes - could be raw SPZ or GZIP-wrapped
    uint8_t first_bytes[4];
    file->get_buffer(first_bytes, 4);

    // Check for GZIP magic (0x1F 0x8B) - file is fully compressed SPZ
    if (first_bytes[0] == 0x1F && first_bytes[1] == 0x8B) {
        // Assume .spz extension means it's an SPZ file
        return p_path.to_lower().ends_with(".spz");
    }

    // Check for raw SPZ magic
    uint32_t magic = first_bytes[0] | (first_bytes[1] << 8) | (first_bytes[2] << 16) | (first_bytes[3] << 24);
    return magic == SPZ_MAGIC;
}

Error SPZLoader::load_file(const String &p_path) {
    GS_LOG_STREAMING_INFO(vformat("[SPZ-LOAD] START: %s", p_path));

    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
    if (file.is_null()) {
        GS_LOG_ERROR_DEFAULT("Failed to open SPZ file: " + p_path);
        return ERR_FILE_NOT_FOUND;
    }

    int64_t file_size = file->get_length();
    GS_LOG_STREAMING_INFO(vformat("[SPZ-LOAD] File opened, size=%d MB", (int)(file_size / 1024 / 1024)));
    if (file_size <= 0 || uint64_t(file_size) > MAX_SPZ_COMPRESSED_BYTES) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ file size exceeds safety cap (%d bytes)", (int64_t)MAX_SPZ_COMPRESSED_BYTES));
        return ERR_FILE_CORRUPT;
    }

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    // Read header (16 bytes)
    if (file_size < (int64_t)sizeof(SPZHeader)) {
        GS_LOG_ERROR_DEFAULT("SPZ file too small: " + p_path);
        return ERR_FILE_CORRUPT;
    }

    // Check if entire file is GZIP compressed (starts with 0x1F 0x8B)
    uint8_t first_bytes[2];
    file->get_buffer(first_bytes, 2);
    file->seek(0);
    GS_LOG_STREAMING_DEBUG(vformat("[SPZ-LOAD] First bytes: 0x%02X 0x%02X (GZIP magic is 0x1F 0x8B)", first_bytes[0], first_bytes[1]));

    PackedByteArray file_data;
    if (first_bytes[0] == 0x1F && first_bytes[1] == 0x8B) {
        // Entire file is GZIP compressed - decompress first
        GS_LOG_STREAMING_DEBUG("[SPZ-LOAD] Detected GZIP, decompressing...");

        PackedByteArray compressed_file;
        compressed_file.resize(file_size);
        file->get_buffer(compressed_file.ptrw(), file_size);

        GS_LOG_STREAMING_DEBUG(vformat("[SPZ-LOAD] Calling decompress_data with %d bytes", (int)compressed_file.size()));
        Error decomp_err = decompress_data(compressed_file, file_data, MAX_SPZ_DECOMPRESSED_BYTES);
        if (decomp_err != OK) {
            GS_LOG_ERROR_DEFAULT("Failed to decompress GZIP-wrapped SPZ file");
            return decomp_err;
        }

        GS_LOG_STREAMING_DEBUG(vformat("[SPZ-LOAD] Decompressed to %d bytes", (int)file_data.size()));

        if (file_data.size() < (int)sizeof(SPZHeader)) {
            GS_LOG_ERROR_DEFAULT("Decompressed SPZ data too small");
            return ERR_FILE_CORRUPT;
        }

        // Parse header from decompressed data
        const uint8_t *hdr = file_data.ptr();
        header.magic = hdr[0] | (hdr[1] << 8) | (hdr[2] << 16) | (hdr[3] << 24);
        header.version = hdr[4] | (hdr[5] << 8) | (hdr[6] << 16) | (hdr[7] << 24);
        header.num_points = hdr[8] | (hdr[9] << 8) | (hdr[10] << 16) | (hdr[11] << 24);
        header.sh_degree = hdr[12];
        header.fractional_bits = hdr[13];
        header.flags = hdr[14];
        header.reserved = hdr[15];

        GS_LOG_STREAMING_DEBUG(vformat("[SPZ-LOAD] Header: magic=0x%08X version=%d points=%d sh=%d frac=%d",
                header.magic, header.version, header.num_points, header.sh_degree, header.fractional_bits));
    } else {
        // Standard SPZ format - header is uncompressed
        header.magic = file->get_32();
        header.version = file->get_32();
        header.num_points = file->get_32();
        header.sh_degree = file->get_8();
        header.fractional_bits = file->get_8();
        header.flags = file->get_8();
        header.reserved = file->get_8();
    }

    // Validate magic number
    if (header.magic != SPZ_MAGIC) {
        GS_LOG_ERROR_DEFAULT(vformat("Invalid SPZ magic number: 0x%08X (expected 0x%08X)", header.magic, SPZ_MAGIC));
        return ERR_FILE_UNRECOGNIZED;
    }

    // Validate version
    if (header.version != SPZ_VERSION_2 && header.version != SPZ_VERSION_3) {
        GS_LOG_ERROR_DEFAULT(vformat("Unsupported SPZ version: %d (supported: 2, 3)", header.version));
        return ERR_FILE_UNRECOGNIZED;
    }

    // Validate reserved field
    if (header.reserved != 0) {
        GS_LOG_STREAMING_WARN("SPZ reserved field is non-zero, file may be from a newer version");
    }

    // Validate SH degree
    if (header.sh_degree > 3) {
        GS_LOG_ERROR_DEFAULT(vformat("Invalid SPZ SH degree: %d (max: 3)", header.sh_degree));
        return ERR_FILE_CORRUPT;
    }

    // Validate point count
    if (header.num_points == 0) {
        GS_LOG_ERROR_DEFAULT("SPZ file contains no points");
        return ERR_FILE_CORRUPT;
    }
    if (header.num_points > MAX_SPZ_POINTS) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ point count exceeds safety cap: %d", header.num_points));
        return ERR_FILE_CORRUPT;
    }

    uint64_t expected_payload_bytes = 0;
    if (!_compute_expected_payload_bytes(header, expected_payload_bytes)) {
        GS_LOG_ERROR_DEFAULT("SPZ payload size computation overflow or invalid header");
        return ERR_FILE_CORRUPT;
    }
    if (expected_payload_bytes == 0 || expected_payload_bytes > MAX_SPZ_DECOMPRESSED_BYTES) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ payload size exceeds safety cap: %d bytes", expected_payload_bytes));
        return ERR_FILE_CORRUPT;
    }
    if (expected_payload_bytes > uint64_t(std::numeric_limits<uint32_t>::max())) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ payload size exceeds parser limits: %d bytes", expected_payload_bytes));
        return ERR_FILE_CORRUPT;
    }

    GS_LOG_STREAMING_INFO(vformat("SPZ header: version=%d, points=%d, sh_degree=%d, fractional_bits=%d, flags=0x%02X",
            header.version, header.num_points, header.sh_degree, header.fractional_bits, header.flags));

    // Get the payload data (after header)
    PackedByteArray decompressed_data;
    const uint8_t *data = nullptr;
    uint32_t data_size = 0;
    uint32_t offset = 0;

    if (file_data.size() > 0) {
        // File was fully GZIP-compressed - payload is already decompressed after header
        const uint64_t payload_size = uint64_t(file_data.size()) - uint64_t(sizeof(SPZHeader));
        if (payload_size != expected_payload_bytes) {
            GS_LOG_ERROR_DEFAULT(vformat("SPZ payload size mismatch: expected %d bytes, got %d bytes",
                    expected_payload_bytes, payload_size));
            return ERR_FILE_CORRUPT;
        }
        data = file_data.ptr() + sizeof(SPZHeader);
        data_size = uint32_t(payload_size);
        GS_LOG_STREAMING_INFO(vformat("SPZ using pre-decompressed data: %d bytes payload", data_size));
    } else {
        // Standard SPZ format - read and decompress the GZIP payload
        uint64_t data_start = file->get_position();
        uint64_t compressed_size = file->get_length() - data_start;

        PackedByteArray compressed_data;
        compressed_data.resize(compressed_size);
        file->get_buffer(compressed_data.ptrw(), compressed_size);

        // Decompress data using gzip
        Error decomp_err = decompress_data(compressed_data, decompressed_data,
                expected_payload_bytes, expected_payload_bytes);
        if (decomp_err != OK) {
            GS_LOG_ERROR_DEFAULT("Failed to decompress SPZ data");
            return decomp_err;
        }

        GS_LOG_STREAMING_INFO(vformat("SPZ decompressed: %d bytes -> %d bytes (%.1fx)",
                compressed_size, decompressed_data.size(),
                (float)decompressed_data.size() / (float)compressed_size));

        data = decompressed_data.ptr();
        data_size = decompressed_data.size();
    }

    if (uint64_t(data_size) != expected_payload_bytes) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ payload size mismatch: expected %d bytes, got %d bytes",
                expected_payload_bytes, data_size));
        return ERR_FILE_CORRUPT;
    }

    // Allocate storage
    LocalVector<Vector3> positions;
    LocalVector<float> alphas;
    LocalVector<Color> colors;
    LocalVector<Vector3> scales;
    LocalVector<Quaternion> rotations;
    LocalVector<float> sh_coeffs;
    uint32_t sh_float_count_per_gaussian = 0;

    positions.resize(header.num_points);
    alphas.resize(header.num_points);
    colors.resize(header.num_points);
    scales.resize(header.num_points);
    rotations.resize(header.num_points);

    // Parse data in order: positions -> alphas -> colors -> scales -> rotations -> SH
    GS_LOG_STREAMING_DEBUG("[SPZ-LOAD] Parsing positions...");
    Error err = parse_positions(data, data_size, offset, positions);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to parse SPZ positions");
        return err;
    }

    GS_LOG_STREAMING_DEBUG("[SPZ-LOAD] Parsing alphas...");
    err = parse_alphas(data, data_size, offset, alphas);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to parse SPZ alphas");
        return err;
    }

    GS_LOG_STREAMING_DEBUG("[SPZ-LOAD] Parsing colors...");
    err = parse_colors(data, data_size, offset, colors);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to parse SPZ colors");
        return err;
    }

    GS_LOG_STREAMING_DEBUG("[SPZ-LOAD] Parsing scales...");
    err = parse_scales(data, data_size, offset, scales);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to parse SPZ scales");
        return err;
    }

    GS_LOG_STREAMING_DEBUG("[SPZ-LOAD] Parsing rotations...");
    if (header.version == SPZ_VERSION_2) {
        err = parse_rotations_v2(data, data_size, offset, rotations);
    } else {
        err = parse_rotations_v3(data, data_size, offset, rotations);
    }
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to parse SPZ rotations");
        return err;
    }

    if (header.sh_degree > 0) {
        GS_LOG_STREAMING_DEBUG("[SPZ-LOAD] Parsing SH...");
        err = parse_spherical_harmonics(data, data_size, offset, sh_coeffs, sh_float_count_per_gaussian);
        if (err != OK) {
            GS_LOG_ERROR_DEFAULT("Failed to parse SPZ spherical harmonics");
            return err;
        }
    }

    GS_LOG_STREAMING_DEBUG("[SPZ-LOAD] Populating GaussianData...");
    // Populate GaussianData
    gaussian_data->resize(header.num_points);

    for (uint32_t i = 0; i < header.num_points; i++) {
        Gaussian g;

        // Position
        g.position = positions[i];

        // Opacity (alpha)
        g.opacity = alphas[i];

        // Color (stored in sh_dc for consistency with PLY loader)
        g.sh_dc = colors[i];

        // Scale (already converted from log scale)
        g.scale = scales[i];

        // Rotation
        g.rotation = rotations[i];
        g.rotation.normalize();

        // Initialize other fields
        g.normal = Vector3(0, 0, 1);
        g.area = 1.0f;
        g.brush_axes = Vector2(1.0f, 1.0f);
        g.stroke_age = 0.0f;
        g.painterly_meta = gaussian_pack_painterly_meta(0);

        // Initialize first-order SH to zero
        for (int j = 0; j < 3; j++) {
            g.sh_1[j] = Vector3();
        }

        gaussian_data->set_gaussian(i, g);

        // Set spherical harmonics if available
        if (sh_float_count_per_gaussian > 0) {
            // Copy correct DC values into sh_coeffs (they were set to 0 as placeholder)
            uint32_t base = i * sh_float_count_per_gaussian;
            sh_coeffs[base + 0] = colors[i].r;
            sh_coeffs[base + 1] = colors[i].g;
            sh_coeffs[base + 2] = colors[i].b;

            const float *sh_ptr = sh_coeffs.ptr() + base;
            gaussian_data->set_spherical_harmonics(i, sh_ptr, sh_float_count_per_gaussian);
        }
    }

    if (offset != data_size) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ payload parser did not consume all bytes (%d/%d)", offset, data_size));
        return ERR_FILE_CORRUPT;
    }

    uint64_t load_time = OS::get_singleton()->get_ticks_usec() - start_time;
    GS_LOG_STREAMING_INFO(vformat("SPZ loaded: %d splats in %.2f ms", header.num_points, load_time / 1000.0));

    return OK;
}

Error SPZLoader::decompress_data(const PackedByteArray &p_compressed, PackedByteArray &r_decompressed,
        uint64_t p_max_decompressed_bytes, uint64_t p_expected_decompressed_bytes) {
    // SPZ uses gzip compression
    // Godot's Compression class supports DEFLATE mode which is the core of gzip
    if (p_compressed.is_empty() || uint64_t(p_compressed.size()) > MAX_SPZ_COMPRESSED_BYTES) {
        GS_LOG_ERROR_DEFAULT("SPZ compressed payload exceeds safety caps");
        return ERR_FILE_CORRUPT;
    }
    if (p_max_decompressed_bytes == 0 || p_max_decompressed_bytes > MAX_SPZ_DECOMPRESSED_BYTES) {
        GS_LOG_ERROR_DEFAULT("SPZ decompression cap is invalid");
        return ERR_FILE_CORRUPT;
    }
    if (p_expected_decompressed_bytes > 0 && p_expected_decompressed_bytes > p_max_decompressed_bytes) {
        GS_LOG_ERROR_DEFAULT("SPZ expected decompressed size exceeds configured cap");
        return ERR_FILE_CORRUPT;
    }

    // gzip format: 10-byte header + compressed data + 8-byte trailer
    // The compressed data is DEFLATE format
    // We need to strip the gzip header/trailer and decompress

    const uint8_t *data = p_compressed.ptr();
    int64_t data_size = p_compressed.size();

    // Check gzip magic number
    if (data_size < 18) { // Minimum gzip: 10 header + 0 data + 8 trailer
        GS_LOG_ERROR_DEFAULT("SPZ compressed data too small for gzip format");
        return ERR_FILE_CORRUPT;
    }

    if (data[0] != 0x1F || data[1] != 0x8B) {
        GS_LOG_ERROR_DEFAULT(vformat("Invalid gzip magic: 0x%02X 0x%02X (expected 0x1F 0x8B)", data[0], data[1]));
        return ERR_FILE_CORRUPT;
    }

    // Check compression method (should be 8 = DEFLATE)
    if (data[2] != 8) {
        GS_LOG_ERROR_DEFAULT(vformat("Unsupported gzip compression method: %d", data[2]));
        return ERR_FILE_CORRUPT;
    }

    uint8_t flags = data[3];
    if ((flags & 0xE0) != 0) {
        GS_LOG_ERROR_DEFAULT(vformat("Invalid gzip flags (reserved bits set): 0x%02X", flags));
        return ERR_FILE_CORRUPT;
    }
    const int64_t trailer_start = data_size - 8;
    int64_t header_end = 10;

    // Skip extra field if present
    if (flags & 0x04) {
        if (header_end + 2 > trailer_start) {
            return ERR_FILE_CORRUPT;
        }
        uint16_t extra_len = data[header_end] | (data[header_end + 1] << 8);
        const int64_t extra_end = header_end + 2 + int64_t(extra_len);
        if (extra_end > trailer_start) {
            GS_LOG_ERROR_DEFAULT("Invalid gzip extra field length");
            return ERR_FILE_CORRUPT;
        }
        header_end = extra_end;
    }

    // Skip filename if present (null-terminated)
    if (flags & 0x08) {
        while (header_end < trailer_start && data[header_end] != 0) {
            header_end++;
        }
        if (header_end >= trailer_start) {
            GS_LOG_ERROR_DEFAULT("Malformed gzip filename field (missing terminator)");
            return ERR_FILE_CORRUPT;
        }
        header_end++; // Skip null terminator
    }

    // Skip comment if present (null-terminated)
    if (flags & 0x10) {
        while (header_end < trailer_start && data[header_end] != 0) {
            header_end++;
        }
        if (header_end >= trailer_start) {
            GS_LOG_ERROR_DEFAULT("Malformed gzip comment field (missing terminator)");
            return ERR_FILE_CORRUPT;
        }
        header_end++; // Skip null terminator
    }

    // Skip CRC16 if present
    if (flags & 0x02) {
        if (header_end + 2 > trailer_start) {
            GS_LOG_ERROR_DEFAULT("Malformed gzip header CRC field");
            return ERR_FILE_CORRUPT;
        }
        header_end += 2;
    }

    if (header_end >= trailer_start) {
        GS_LOG_ERROR_DEFAULT("Invalid gzip structure");
        return ERR_FILE_CORRUPT;
    }

    // Extract DEFLATE data (everything between header and 8-byte trailer)
    int64_t deflate_size = trailer_start - header_end;
    if (deflate_size <= 0) {
        GS_LOG_ERROR_DEFAULT("No compressed data in gzip stream");
        return ERR_FILE_CORRUPT;
    }

    // Read original size from gzip trailer (last 4 bytes, little-endian)
    uint32_t original_size = data[data_size - 4] |
            (data[data_size - 3] << 8) |
            (data[data_size - 2] << 16) |
            (data[data_size - 1] << 24);
    const uint64_t declared_size = uint64_t(original_size);
    if (declared_size == 0) {
        GS_LOG_ERROR_DEFAULT("SPZ gzip trailer declares zero decompressed bytes");
        return ERR_FILE_CORRUPT;
    }
    if (declared_size > p_max_decompressed_bytes || declared_size > MAX_SPZ_DECOMPRESSED_BYTES) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ declared decompressed size exceeds safety cap: %d bytes", declared_size));
        return ERR_FILE_CORRUPT;
    }
    if (p_expected_decompressed_bytes > 0 && declared_size != p_expected_decompressed_bytes) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ gzip trailer size mismatch: expected %d bytes, got %d bytes",
                p_expected_decompressed_bytes, declared_size));
        return ERR_FILE_CORRUPT;
    }
    if (declared_size > uint64_t(std::numeric_limits<int>::max())) {
        GS_LOG_ERROR_DEFAULT(vformat("SPZ declared decompressed size exceeds allocator limits: %d bytes",
                declared_size));
        return ERR_FILE_CORRUPT;
    }
    const int expected_size = int(declared_size);

    GS_LOG_STREAMING_DEBUG(vformat("[SPZ-DECOMP] compressed=%d MB, deflate=%d MB, original_size=%d MB",
            (int)(data_size / 1024 / 1024), (int)(deflate_size / 1024 / 1024), expected_size / 1024 / 1024));

    r_decompressed.resize(expected_size);

    // Strategy 1: let engine gzip mode handle complete stream.
    int result = Compression::decompress(r_decompressed.ptrw(), expected_size,
            p_compressed.ptr(), p_compressed.size(), Compression::MODE_GZIP);
    if (result == expected_size) {
        return OK;
    }

    // Strategy 2: strip gzip framing and decode raw DEFLATE.
    PackedByteArray deflate_data;
    deflate_data.resize(deflate_size);
    memcpy(deflate_data.ptrw(), data + header_end, deflate_size);
    result = Compression::decompress(r_decompressed.ptrw(), expected_size,
            deflate_data.ptr(), deflate_size, Compression::MODE_DEFLATE);
    if (result == expected_size) {
        return OK;
    }

    GS_LOG_ERROR_DEFAULT(vformat("Failed to decompress SPZ payload deterministically (result=%d expected=%d)",
            result, expected_size));
    return ERR_FILE_CORRUPT;
}

Error SPZLoader::parse_positions(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Vector3> &r_positions) {
    // Positions: 3 x 24-bit fixed-point signed integers per point
    const uint64_t needed = uint64_t(header.num_points) * 9ull;
    ERR_FAIL_COND_V(!_offset_range_valid(r_offset, needed, p_data_size), ERR_FILE_CORRUPT);
    for (uint32_t i = 0; i < header.num_points; i++) {
        int32_t x = read_int24(p_data + r_offset);
        int32_t y = read_int24(p_data + r_offset + 3);
        int32_t z = read_int24(p_data + r_offset + 6);
        r_offset += 9;

        r_positions[i] = Vector3(
                fixed_to_float(x, header.fractional_bits),
                fixed_to_float(y, header.fractional_bits),
                fixed_to_float(z, header.fractional_bits));
    }

    return OK;
}

Error SPZLoader::parse_alphas(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<float> &r_alphas) {
    // Alphas: 1 x 8-bit unsigned integer per point
    ERR_FAIL_COND_V(!_offset_range_valid(r_offset, header.num_points, p_data_size), ERR_FILE_CORRUPT);
    for (uint32_t i = 0; i < header.num_points; i++) {
        r_alphas[i] = decode_alpha(p_data[r_offset++]);
    }

    return OK;
}

Error SPZLoader::parse_colors(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Color> &r_colors) {
    // Colors: 3 x 8-bit unsigned integers (RGB) per point
    // SPZ stores linear RGB [0-255] representing the DC contribution directly.
    // Keep values in 0-1 space and do NOT divide by SH_C0 (that would yield coefficients).
    const uint64_t needed = uint64_t(header.num_points) * 3ull;
    ERR_FAIL_COND_V(!_offset_range_valid(r_offset, needed, p_data_size), ERR_FILE_CORRUPT);

    for (uint32_t i = 0; i < header.num_points; i++) {
        // Read as linear RGB [0-1]
        float r = p_data[r_offset++] / 255.0f;
        float g = p_data[r_offset++] / 255.0f;
        float b = p_data[r_offset++] / 255.0f;

        r_colors[i] = Color(r, g, b, 1.0f);
    }

    return OK;
}

Error SPZLoader::parse_scales(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Vector3> &r_scales) {
    // Scales: 3 x 8-bit log-encoded values per point
    const uint64_t needed = uint64_t(header.num_points) * 3ull;
    ERR_FAIL_COND_V(!_offset_range_valid(r_offset, needed, p_data_size), ERR_FILE_CORRUPT);
    for (uint32_t i = 0; i < header.num_points; i++) {
        float sx = decode_scale(p_data[r_offset++]);
        float sy = decode_scale(p_data[r_offset++]);
        float sz = decode_scale(p_data[r_offset++]);
        r_scales[i] = Vector3(sx, sy, sz);
    }

    return OK;
}

Error SPZLoader::parse_rotations_v2(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Quaternion> &r_rotations) {
    // Version 2: (x, y, z) quaternion components as 8-bit signed integers
    // w is computed from normalization
    // SPZ uses RUB, Godot uses RUF - apply Z-flip transformation to quaternion
    const uint64_t needed = uint64_t(header.num_points) * 3ull;
    ERR_FAIL_COND_V(!_offset_range_valid(r_offset, needed, p_data_size), ERR_FILE_CORRUPT);
    for (uint32_t i = 0; i < header.num_points; i++) {
        int8_t qx = (int8_t)p_data[r_offset++];
        int8_t qy = (int8_t)p_data[r_offset++];
        int8_t qz = (int8_t)p_data[r_offset++];

        // Convert from [-127, 127] to [-1, 1]
        float x = qx / 127.0f;
        float y = qy / 127.0f;
        float z = qz / 127.0f;

        // Compute w (assume positive w)
        float sum_sq = x * x + y * y + z * z;
        float w = 1.0f;
        if (sum_sq < 1.0f) {
            w = sqrtf(1.0f - sum_sq);
        } else {
            // Normalize if sum exceeds 1
            float scale = 1.0f / sqrtf(sum_sq);
            x *= scale;
            y *= scale;
            z *= scale;
            w = 0.0f;
        }

        r_rotations[i] = Quaternion(x, y, z, w);
    }

    return OK;
}

Error SPZLoader::parse_rotations_v3(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset, LocalVector<Quaternion> &r_rotations) {
    // Version 3: Smallest-three encoding (matching Niantic's unpackQuaternionSmallestThree)
    // Bit layout: [2-bit largest_idx][10-bit comp2][10-bit comp1][10-bit comp0]
    // Each 10-bit component: 9-bit unsigned magnitude + 1 sign bit
    const float sqrt1_2 = 0.7071067811865476f;
    const uint32_t c_mask = (1u << 9u) - 1u;  // 9-bit magnitude mask = 511
    const uint64_t needed = uint64_t(header.num_points) * 4ull;
    ERR_FAIL_COND_V(!_offset_range_valid(r_offset, needed, p_data_size), ERR_FILE_CORRUPT);

    for (uint32_t i = 0; i < header.num_points; i++) {
        // Read 4 bytes as little-endian uint32
        uint32_t comp = p_data[r_offset] |
                (p_data[r_offset + 1] << 8) |
                (p_data[r_offset + 2] << 16) |
                (p_data[r_offset + 3] << 24);
        r_offset += 4;

        // Extract largest component index (bits 30-31)
        int i_largest = comp >> 30;

        // Extract the three smallest components (Niantic iterates i=3 down to 0)
        float rotation[4] = { 0, 0, 0, 0 };
        float sum_squares = 0;

        for (int k = 3; k >= 0; --k) {
            if (k != i_largest) {
                uint32_t mag = comp & c_mask;        // 9-bit unsigned magnitude
                uint32_t negbit = (comp >> 9u) & 0x1u;  // sign bit
                comp = comp >> 10u;                   // shift to next component

                rotation[k] = sqrt1_2 * ((float)mag) / (float)c_mask;
                if (negbit == 1) {
                    rotation[k] = -rotation[k];
                }
                sum_squares += rotation[k] * rotation[k];
            }
        }

        // Reconstruct the largest component (always positive)
        // Clamp to handle floating point precision issues
        float largest_sq = MAX(0.0f, 1.0f - sum_squares);
        rotation[i_largest] = sqrtf(largest_sq);

        // rotation[0]=x, rotation[1]=y, rotation[2]=z, rotation[3]=w
        Quaternion q(rotation[0], rotation[1], rotation[2], rotation[3]);
        q.normalize();  // Ensure unit quaternion for numerical stability
        r_rotations[i] = q;
    }

    return OK;
}

Error SPZLoader::parse_spherical_harmonics(const uint8_t *p_data, uint32_t p_data_size, uint32_t &r_offset,
        LocalVector<float> &r_sh_coeffs, uint32_t &r_sh_float_count_per_gaussian) {
    // SH coefficients are stored per-channel (RGB), with variable bit precision
    // Degree 0: 3 coefficients per gaussian (DC term only) - but this is in colors
    // Degree 1: 9 coefficients (3 per color channel, 3 directions)
    // Degree 2: 24 coefficients
    // Degree 3: 45 coefficients

    if (header.sh_degree == 0) {
        r_sh_float_count_per_gaussian = 3; // Just DC term (already in colors)
        return OK;
    }

    // Number of SH coefficients per gaussian (excluding DC which is in colors)
    uint32_t sh_count = SH_COEFFS_PER_DEGREE[header.sh_degree];
    r_sh_float_count_per_gaussian = 3 + sh_count; // DC (3) + rest

    const uint64_t total_coeffs = uint64_t(header.num_points) * uint64_t(r_sh_float_count_per_gaussian);
    ERR_FAIL_COND_V(total_coeffs > uint64_t(std::numeric_limits<int>::max()), ERR_FILE_CORRUPT);
    r_sh_coeffs.resize((int)total_coeffs);

    // The SH data is organized with color channels as inner axis
    // First degree 0 (DC) - already parsed in colors
    // Then degree 1+ coefficients

    for (uint32_t i = 0; i < header.num_points; i++) {
        uint32_t out_base = i * r_sh_float_count_per_gaussian;

        // DC term - placeholder (actual DC comes from colors)
        r_sh_coeffs[out_base + 0] = 0.0f;
        r_sh_coeffs[out_base + 1] = 0.0f;
        r_sh_coeffs[out_base + 2] = 0.0f;
    }

    // Parse higher-order SH coefficients
    // They're stored sequentially: all coeffs for point 0, then all for point 1, etc.
    // Within each point: coefficients organized by band, then by color channel

    uint32_t sh_rest_count = sh_count;
    if (sh_rest_count > 0) {
        const uint64_t needed = uint64_t(header.num_points) * uint64_t(sh_rest_count);
        ERR_FAIL_COND_V(!_offset_range_valid(r_offset, needed, p_data_size), ERR_FILE_CORRUPT);
        for (uint32_t i = 0; i < header.num_points; i++) {
            uint32_t out_base = i * r_sh_float_count_per_gaussian + 3; // Skip DC

            for (uint32_t j = 0; j < sh_rest_count; j++) {
                uint8_t bits = (j < 9) ? SH_BITS_DEGREE_0 : SH_BITS_DEGREE_1_2;
                uint8_t encoded = p_data[r_offset++];
                r_sh_coeffs[out_base + j] = decode_sh_coefficient(encoded, bits);
            }
        }
    }

    return OK;
}

float SPZLoader::fixed_to_float(int32_t p_fixed, uint8_t p_fractional_bits) const {
    return (float)p_fixed / (float)(1 << p_fractional_bits);
}

int32_t SPZLoader::read_int24(const uint8_t *p_data) const {
    // Read 24-bit little-endian signed integer
    int32_t value = p_data[0] | (p_data[1] << 8) | (p_data[2] << 16);

    // Sign extend if negative (bit 23 set)
    if (value & 0x800000) {
        value |= 0xFF000000; // Extend sign bit
    }

    return value;
}

float SPZLoader::decode_scale(uint8_t p_encoded) const {
    // Scale is stored as log scale, quantized to 8 bits
    // Niantic formula: log_scale = encoded/16.0 - 10.0
    // Then we apply exp() to get linear scale (same as PLY loader)
    float log_scale = p_encoded / 16.0f - 10.0f;
    return expf(log_scale);
}

float SPZLoader::decode_alpha(uint8_t p_encoded) const {
    // Alpha is stored as sigmoid, we need to apply inverse sigmoid
    // But for SPZ, alpha appears to be direct [0, 255] -> [0, 1]
    // Let's check if it needs logit conversion

    // Based on research, SPZ stores alpha directly as linear [0, 255] -> [0, 1]
    return p_encoded / 255.0f;
}

float SPZLoader::decode_sh_coefficient(uint8_t p_encoded, uint8_t /* p_bits */) const {
    // SPZ stores SH coefficients as 8-bit values centered at 128
    // This matches the official Niantic unquantizeSH() function:
    //   float unquantizeSH(uint8_t x) { return (x - 128.0f) / 128.0f; }
    // The p_bits parameter was incorrectly used before - SPZ always uses
    // full 8-bit range with 128 as zero point, regardless of "bit precision"
    // mentioned in the spec (which refers to internal quantization, not file format)
    return (static_cast<float>(p_encoded) - 128.0f) / 128.0f;
}

int SPZLoader::get_splat_count() const {
    return gaussian_data.is_valid() ? gaussian_data->get_count() : 0;
}

Dictionary SPZLoader::get_load_statistics() const {
    Dictionary stats;
    stats["splat_count"] = get_splat_count();
    stats["format"] = "spz";
    stats["version"] = (int)header.version;
    stats["sh_degree"] = (int)header.sh_degree;
    stats["fractional_bits"] = (int)header.fractional_bits;
    stats["flags"] = (int)header.flags;
    stats["antialiased"] = (header.flags & SPZ_FLAG_ANTIALIASED) != 0;

    if (gaussian_data.is_valid()) {
        AABB aabb = gaussian_data->get_aabb();
        stats["bounds_min"] = aabb.position;
        stats["bounds_max"] = aabb.position + aabb.size;
    }

    return stats;
}
