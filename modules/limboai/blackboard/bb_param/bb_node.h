/**
 * bb_node.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_NODE_H
#define BB_NODE_H

#include "bb_param.h"

class BBNode : public BBParam {
	GDCLASS(BBNode, BBParam);

protected:
	virtual Variant::Type get_type() const override { return Variant::NODE_PATH; }

public:
	virtual Variant get_value(Object *p_agent, const Ref<Blackboard> &p_blackboard, const Variant &p_default = Variant()) override;
};

#endif // BB_NODE_H
