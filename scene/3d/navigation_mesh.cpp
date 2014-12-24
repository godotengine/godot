#include "navigation_mesh.h"
#include "navigation.h"


void NavigationMesh::create_from_mesh(const Ref<Mesh>& p_mesh) {


	vertices=DVector<Vector3>();
	clear_polygons();

	for(int i=0;i<p_mesh->get_surface_count();i++) {

		if (p_mesh->surface_get_primitive_type(i)!=Mesh::PRIMITIVE_TRIANGLES)
			continue;
		Array arr = p_mesh->surface_get_arrays(i);
		DVector<Vector3> varr = arr[Mesh::ARRAY_VERTEX];
		DVector<int> iarr = arr[Mesh::ARRAY_INDEX];
		if (varr.size()==0 || iarr.size()==0)
			continue;

		int from = vertices.size();
		vertices.append_array(varr);
		int rlen = iarr.size();
		DVector<int>::Read r = iarr.read();

		for(int j=0;j<rlen;j+=3) {
			Vector<int> vi;
			vi.resize(3);
			vi[0]=r[j+0]+from;
			vi[1]=r[j+1]+from;
			vi[2]=r[j+2]+from;

			add_polygon(vi);
		}
	}
}

void NavigationMesh::set_vertices(const DVector<Vector3>& p_vertices) {

	vertices=p_vertices;
}

DVector<Vector3> NavigationMesh::get_vertices() const{

	return vertices;
}


void NavigationMesh::_set_polygons(const Array& p_array) {

	polygons.resize(p_array.size());
	for(int i=0;i<p_array.size();i++) {
		polygons[i].indices=p_array[i];
	}
}

Array NavigationMesh::_get_polygons() const {

	Array ret;
	ret.resize(polygons.size());
	for(int i=0;i<ret.size();i++) {
		ret[i]=polygons[i].indices;
	}

	return ret;
}


void NavigationMesh::add_polygon(const Vector<int>& p_polygon){

	Polygon polygon;
	polygon.indices=p_polygon;
	polygons.push_back(polygon);

}
int NavigationMesh::get_polygon_count() const{

	return polygons.size();
}
Vector<int> NavigationMesh::get_polygon(int p_idx){

	ERR_FAIL_INDEX_V(p_idx,polygons.size(),Vector<int>());
	return polygons[p_idx].indices;
}
void NavigationMesh::clear_polygons(){

	polygons.clear();
}

void NavigationMesh::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_vertices","vertices"),&NavigationMesh::set_vertices);
	ObjectTypeDB::bind_method(_MD("get_vertices"),&NavigationMesh::get_vertices);

	ObjectTypeDB::bind_method(_MD("add_polygon","polygon"),&NavigationMesh::add_polygon);
	ObjectTypeDB::bind_method(_MD("get_polygon_count"),&NavigationMesh::get_polygon_count);
	ObjectTypeDB::bind_method(_MD("get_polygon","idx"),&NavigationMesh::get_polygon);
	ObjectTypeDB::bind_method(_MD("clear_polygons"),&NavigationMesh::clear_polygons);

	ObjectTypeDB::bind_method(_MD("_set_polygons","polygons"),&NavigationMesh::_set_polygons);
	ObjectTypeDB::bind_method(_MD("_get_polygons"),&NavigationMesh::_get_polygons);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3_ARRAY,"vertices",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("set_vertices"),_SCS("get_vertices"));
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"polygons",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("_set_polygons"),_SCS("_get_polygons"));
}

NavigationMesh::NavigationMesh() {

}

void NavigationMeshInstance::set_enabled(bool p_enabled) {

	if (enabled==p_enabled)
		return;
	enabled=p_enabled;

	if (!is_inside_tree())
		return;

	if (!enabled) {

		if (nav_id!=-1) {
			navigation->navmesh_remove(nav_id);
			nav_id=-1;
		}
	} else {

		if (navigation) {

			if (navmesh.is_valid()) {

				nav_id = navigation->navmesh_create(navmesh,get_relative_transform(navigation),this);
			}
		}

	}

	update_gizmo();
}

bool NavigationMeshInstance::is_enabled() const {


	return enabled;
}


/////////////////////////////


void NavigationMeshInstance::_notification(int p_what) {


	switch(p_what) {
		case NOTIFICATION_ENTER_TREE: {

			Spatial *c=this;
			while(c) {

				navigation=c->cast_to<Navigation>();
				if (navigation) {

					if (enabled && navmesh.is_valid()) {

						nav_id = navigation->navmesh_create(navmesh,get_relative_transform(navigation),this);
					}
					break;
				}

				c=c->get_parent_spatial();
			}

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (navigation && nav_id!=-1) {
				navigation->navmesh_set_transform(nav_id,get_relative_transform(navigation));
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (navigation) {

				if (nav_id!=-1) {
					navigation->navmesh_remove(nav_id);
					nav_id=-1;
				}
			}
			navigation=NULL;
		} break;
	}
}


void NavigationMeshInstance::set_navigation_mesh(const Ref<NavigationMesh>& p_navmesh) {

	if (p_navmesh==navmesh)
		return;

	if (navigation && nav_id!=-1) {
		navigation->navmesh_remove(nav_id);
		nav_id=-1;
	}
	navmesh=p_navmesh;

	if (navigation && navmesh.is_valid() && enabled) {
		nav_id = navigation->navmesh_create(navmesh,get_relative_transform(navigation),this);
	}
	update_gizmo();

}

Ref<NavigationMesh> NavigationMeshInstance::get_navigation_mesh() const{

	return navmesh;
}

void NavigationMeshInstance::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_navigation_mesh","navmesh"),&NavigationMeshInstance::set_navigation_mesh);
	ObjectTypeDB::bind_method(_MD("get_navigation_mesh"),&NavigationMeshInstance::get_navigation_mesh);

	ObjectTypeDB::bind_method(_MD("set_enabled","enabled"),&NavigationMeshInstance::set_enabled);
	ObjectTypeDB::bind_method(_MD("is_enabled"),&NavigationMeshInstance::is_enabled);

	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"navmesh",PROPERTY_HINT_RESOURCE_TYPE,"NavigationMesh"),_SCS("set_navigation_mesh"),_SCS("get_navigation_mesh"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"enabled"),_SCS("set_enabled"),_SCS("is_enabled"));
}

NavigationMeshInstance::NavigationMeshInstance() {

	navigation=NULL;
	nav_id=-1;
	enabled=true;

}
