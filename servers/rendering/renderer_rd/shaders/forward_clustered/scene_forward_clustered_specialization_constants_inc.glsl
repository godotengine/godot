/* Specialization Constants (Toggles) */

layout(constant_id = 0) const bool sc_use_forward_gi = false;
layout(constant_id = 1) const bool sc_use_light_projector = false;
layout(constant_id = 2) const bool sc_use_light_soft_shadows = false;
layout(constant_id = 3) const bool sc_use_directional_soft_shadows = false;

/* Specialization Constants (Values) */

layout(constant_id = 6) const uint sc_soft_shadow_samples = 4;
layout(constant_id = 7) const uint sc_penumbra_shadow_samples = 4;

layout(constant_id = 8) const uint sc_directional_soft_shadow_samples = 4;
layout(constant_id = 9) const uint sc_directional_penumbra_shadow_samples = 4;

layout(constant_id = 10) const bool sc_decal_use_mipmaps = true;
layout(constant_id = 11) const bool sc_projector_use_mipmaps = true;
layout(constant_id = 12) const bool sc_use_depth_fog = false;
