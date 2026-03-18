#include "gaussian_splat_config_registry.h"

#include "gaussian_splat_settings_manager.h"
#include "../lod/lod_config.h"
#include "../renderer/float16_config.h"
#include "../renderer/gpu_sorting_config.h"
#include "../renderer/pipeline_feature_set.h"
#include "../renderer/quantization_config.h"
#include "../renderer/sh_config.h"

void GaussianSplatConfigRegistry::initialize_all() {
    initialize_gaussian_splat_settings();
    initialize_gpu_sorting_config();
    initialize_float16_config();
    initialize_sh_config();
    initialize_quantization_config();
    initialize_pipeline_feature_set();
    initialize_lod_config();
}
