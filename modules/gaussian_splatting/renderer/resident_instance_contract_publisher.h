#ifndef GS_RESIDENT_INSTANCE_CONTRACT_PUBLISHER_H
#define GS_RESIDENT_INSTANCE_CONTRACT_PUBLISHER_H

#include "core/string/ustring.h"

class GaussianSplatRenderer;

namespace ResidentInstanceContractPublisher {

bool publish(GaussianSplatRenderer *p_renderer, bool p_allow_primary_fallback_instance, String *r_reason = nullptr);

} // namespace ResidentInstanceContractPublisher

#endif // GS_RESIDENT_INSTANCE_CONTRACT_PUBLISHER_H
