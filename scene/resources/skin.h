#ifndef SKIN_H
#define SKIN_H

#include "core/resource.h"

class Skin : public Resource {
	GDCLASS(Skin, Resource)

	struct Bind {
		int bone;
		Transform pose;
	};

	Vector<Bind> binds;

	Bind *binds_ptr;
	int bind_count;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	void set_bind_count(int p_size);
	inline int get_bind_count() const { return bind_count; }

	void add_bind(int p_bone, const Transform &p_pose);

	void set_bind_bone(int p_index, int p_bone);
	void set_bind_pose(int p_index, const Transform &p_pose);

	inline int get_bind_bone(int p_index) const {
#ifdef DEBUG_ENABLED
		ERR_FAIL_INDEX_V(p_index, bind_count, -1);
#endif
		return binds_ptr[p_index].bone;
	}

	inline Transform get_bind_pose(int p_index) const {
#ifdef DEBUG_ENABLED
		ERR_FAIL_INDEX_V(p_index, bind_count, Transform());
#endif
		return binds_ptr[p_index].pose;
	}

	void clear_binds();

	Skin();
};

#endif // SKIN_H
