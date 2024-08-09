/* Spec Constants */

#if !defined(MODE_RENDER_DEPTH)

#if !defined(MODE_UNSHADED)

layout(constant_id = 0) const bool sc_use_light_projector = false;
layout(constant_id = 1) const bool sc_use_light_soft_shadows = false;
layout(constant_id = 2) const bool sc_use_directional_soft_shadows = false;

layout(constant_id = 3) const uint sc_soft_shadow_samples = 4;
layout(constant_id = 4) const uint sc_penumbra_shadow_samples = 4;

layout(constant_id = 5) const uint sc_directional_soft_shadow_samples = 4;
layout(constant_id = 6) const uint sc_directional_penumbra_shadow_samples = 4;

layout(constant_id = 8) const bool sc_projector_use_mipmaps = true;

layout(constant_id = 9) const bool sc_disable_omni_lights = false;
layout(constant_id = 10) const bool sc_disable_spot_lights = false;
layout(constant_id = 11) const bool sc_disable_reflection_probes = false;
layout(constant_id = 12) const bool sc_disable_directional_lights = false;

#endif //!MODE_UNSHADED

layout(constant_id = 7) const bool sc_decal_use_mipmaps = true;
layout(constant_id = 13) const bool sc_disable_decals = false;
layout(constant_id = 14) const bool sc_disable_fog = false;
layout(constant_id = 16) const bool sc_use_depth_fog = false;

#endif //!MODE_RENDER_DEPTH

layout(constant_id = 15) const float sc_luminance_multiplier = 2.0;
layout(constant_id = 17) const bool sc_is_multimesh = false;
