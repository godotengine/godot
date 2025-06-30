#pragma once
#include "gdextension_spx_ext.h"
#include "scene/main/node.h"

class SpxInputProxy : public Node {
	GDCLASS(SpxInputProxy, Node);

public:
	void ready();
protected:
	void input(const Ref<InputEvent> &p_event) override;
};
