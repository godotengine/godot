#pragma once

#include "scene/main/node.h"

#include <memory>

class UnknoterImpl;

class UnknoterNode : public Node {
	GDCLASS(UnknoterNode, Node);

  std::unique_ptr<UnknoterImpl> impl_;
protected:
  static void _bind_methods();

public:
  UnknoterNode();
  ~UnknoterNode();

  int add(int a, int b);
};
