#ifndef SUBSPACE_2D_H
#define SUBSPACE_2D_H

#include "scene/2d/node_2d.h"
#include "scene/resources/space_2d.h"

class Subspace2D : public Node2D {

	GDCLASS(Subspace2D, Node2D)

private:
	Ref<Space2D> space_2d;

protected:
	static void _bind_methods();

public:
	void set_space_2d(const Ref<Space2D> &p_space_2d);
	Ref<Space2D> get_space_2d() const;

	Subspace2D();
	~Subspace2D();
};

#endif // SUBSPACE_2D_H
