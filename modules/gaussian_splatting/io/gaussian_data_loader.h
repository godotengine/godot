#ifndef GAUSSIAN_DATA_LOADER_H
#define GAUSSIAN_DATA_LOADER_H

#include "core/variant/variant.h"
#include "../core/gaussian_data.h"

struct GaussianDataLoadResult {
    Ref<GaussianData> data;
    PackedStringArray missing_required;
    PackedStringArray missing_optional;
    bool used_ply = false;
    bool used_spz = false;
};

Error load_gaussian_data_from_file(const String &p_path, GaussianDataLoadResult &r_result);

#endif // GAUSSIAN_DATA_LOADER_H
