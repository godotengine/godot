#ifndef REMOTETRANSFORM_H
#define REMOTETRANSFORM_H

#include "scene/3d/spatial.h"

class RemoteTransform : public Spatial
{
	GDCLASS(RemoteTransform,Spatial);
	
	NodePath remote_node;

	ObjectID cache;

	void _update_remote();
	void _update_cache(); 
 
protected:
	static void _bind_methods();
	void _notification(int p_what);
public:
	void set_remote_node(const NodePath& p_remote_node);
	NodePath get_remote_node() const;
	
	virtual String get_configuration_warning() const;
	
	RemoteTransform();

};

#endif // REMOTETRANSFORM_H
