#ifndef GAUSSIAN_SPLAT_CONFIG_REGISTRY_H
#define GAUSSIAN_SPLAT_CONFIG_REGISTRY_H

class GaussianSplatConfigRegistry {
public:
    // Keep init order centralized; mirrors prior register_types.cpp sequence.
    static void initialize_all();
};

#endif // GAUSSIAN_SPLAT_CONFIG_REGISTRY_H
