#include "gaussian_data_loader.h"

#include "ply_loader.h"
#include "spz_loader.h"

Error load_gaussian_data_from_file(const String &p_path, GaussianDataLoadResult &r_result) {
    r_result = GaussianDataLoadResult();

    const String extension = p_path.get_extension().to_lower();
    if (extension == "spz") {
        Ref<SPZLoader> spz_loader;
        spz_loader.instantiate();

        Error err = spz_loader->load_file(p_path);
        if (err != OK) {
            return err;
        }

        Ref<::GaussianData> gaussian_data = spz_loader->get_gaussian_data();
        if (gaussian_data.is_null() || gaussian_data->get_count() == 0) {
            return ERR_FILE_CORRUPT;
        }

        r_result.data = gaussian_data;
        r_result.used_spz = true;
        return OK;
    }

    Ref<PLYLoader> ply_loader;
    ply_loader.instantiate();

    Error err = ply_loader->load_file(p_path);
    if (err != OK) {
        return err;
    }

    ply_loader->get_property_deficiencies(r_result.missing_required, r_result.missing_optional);

    Ref<::GaussianData> gaussian_data = ply_loader->get_gaussian_data();
    if (gaussian_data.is_null() || gaussian_data->get_count() == 0) {
        return ERR_FILE_CORRUPT;
    }

    r_result.data = gaussian_data;
    r_result.used_ply = true;
    return OK;
}
