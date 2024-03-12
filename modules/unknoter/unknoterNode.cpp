#include "unknoterNode.h"

class UnknoterImpl {
public:
  int add(int a, int b) {
    return b + a;
  }
};

void UnknoterNode::_bind_methods(){
  ClassDB::bind_method(D_METHOD("add", "a", "b"), &UnknoterNode::add);
}

UnknoterNode::UnknoterNode()
  : impl_(new UnknoterImpl())
{}

UnknoterNode::~UnknoterNode() = default;

int UnknoterNode::add(int a, int b) {
  return impl_->add(a, b);
}
