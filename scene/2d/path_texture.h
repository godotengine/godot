#ifndef PATH_TEXTURE_H
#define PATH_TEXTURE_H

#include "scene/2d/node_2d.h"

class PathTexture : public Node2D {
	OBJ_TYPE( PathTexture, Node2D );

	Ref<Texture> begin;
	Ref<Texture> repeat;
	Ref<Texture> end;
	int subdivs;
	bool overlap;
public:

	void set_begin_texture(const Ref<Texture>& p_texture);
	Ref<Texture> get_begin_texture() const;

	void set_repeat_texture(const Ref<Texture>& p_texture);
	Ref<Texture> get_repeat_texture() const;

	void set_end_texture(const Ref<Texture>& p_texture);
	Ref<Texture> get_end_texture() const;

	void set_subdivisions(int p_amount);
	int get_subdivisions() const;

	void set_overlap(int p_amount);
	int get_overlap() const;

	PathTexture();
};

#endif // PATH_TEXTURE_H
