#include "scene_chunk.h"

#include "scene/3d/physics/collision_object_3d.h"

#include "scene/resources/3d/box_shape_3d.h"
#include "scene/resources/3d/sphere_shape_3d.h"



void SceneDataCompoent::show(int p_show_level,const Transform3D& p_parent,SceneChunkGroupInstance * instance) {

    if(instance == nullptr) {
        return;
    }
    Transform3D trans = p_parent * transform;
    if(resource_type == ResourceType::RT_Mesh) {
        if(!resource_path.is_empty()) {
            instance->_add_mesh_instance(resource_path, trans);
        }        
    }
    if(collision_shape_type == CollisionShapeType::SHAPE_NONE) {
        return;
    }
    else if(collision_shape_type == CollisionShapeType::SHAPE_MESH) {
        if(!collision_mesh_path.is_empty()) {
            instance->_add_mesh_collision_instance(collision_mesh_path, trans);
        }        
    }
    else if(collision_shape_type != CollisionShapeType::SHAPE_SPHERE) {        
        instance->_add_collision_instance(trans,collision_shape_type,collosion_box_size,0,collision_sphere_radius);
    }
    else if(collision_shape_type == CollisionShapeType::SHAPE_BOX) {
        instance->_add_collision_instance(trans,collision_shape_type,collosion_box_size,0,0);
    }
    else if(collision_shape_type == CollisionShapeType::SHAPE_CAPSULE) {
        instance->_add_collision_instance(trans,collision_shape_type,collosion_box_size,collision_capsule_height,collision_capsule_radius);
    }
    else if(collision_shape_type == CollisionShapeType::SHAPE_CYLINDER) {
        instance->_add_collision_instance(trans,collision_shape_type,collosion_box_size,collision_cylinder_height,collision_cylinder_radius);
    }
    
}
/*********************************************************************************************************/
void SceneDataCompoentBlock::show(int p_show_level,const Transform3D& p_parent, SceneChunkGroupInstance * instance) {
	SceneDataCompoent::show(p_show_level,p_parent,instance);
    Transform3D world = p_parent * transform;
    for(int i = 0; i < compoents.size(); i++) {
        if(p_show_level >=compoents[i].second) {
            compoents[i].first->show(p_show_level,world,instance);
        }
    }
}
/*********************************************************************************************************/
void SceneBlock::show(int p_show_level,const Transform3D& p_parent, SceneChunkGroupInstance * instance) {
    Transform3D trans = p_parent * transform;
    for(int i = 0; i < blocks.size(); i++) {
        if(p_show_level >= blocks[i].second) {
			blocks[i].first->show(p_show_level, trans,instance);
        }
    }
}
/*********************************************************************************************************/
void SceneResource::show(int lod,int p_show_level,const Transform3D& p_parent, SceneChunkGroupInstance * instance) {
    int curr_lod = lod;
    if(curr_lod >= scene_lods.size()) {
        curr_lod = scene_lods.size() - 1;
    }
    if(p_show_level >= scene_lods[curr_lod].second) {
        return;
    }
    scene_lods[curr_lod].first->show(p_show_level,p_parent,instance);
}
/*********************************************************************************************************/


void SceneChunk::MeshInstance::update_mesh_instance() {
    if(!dirty) {
        return;
    }
    if(multimesh.is_null()) {
        Ref<Resource> resource;
        if(load_token.is_valid()) {
            if(ResourceLoader::load_threaded_get_status(load_token->local_path,resource) != ResourceLoader::THREAD_LOAD_LOADED) {
                return;
            }
        }
        Ref<Mesh> mesh = resource;
        if(mesh.is_null()) {
            return;
        }
        multimesh = memnew(MultiMesh);
        multimesh->set_mesh(mesh);
        
    }
    dirty = false;
    MultiMeshInstance3D* multimesh_instance = nullptr;
    if (mult_mesh_instances_id.is_valid()) {
        multimesh_instance = Object::cast_to<MultiMeshInstance3D>(ObjectDB::get_instance(node_id));
    } 
    if(multimesh_instance == nullptr) {
        multimesh_instance = memnew(MultiMeshInstance3D);
    }
	if (multimesh == nullptr) {
		multimesh = memnew(MultiMesh);
		multimesh_instance->set_multimesh(multimesh);
	}

    multimesh->set_instance_count(mesh_transforms.size());
    int i = 0;
    mesh_id_maps.clear();
    for(auto& it : mesh_transforms) {
        multimesh->set_instance_transform(i,it.value.transform);
        multimesh->set_instance_color(i,it.value.color);
        multimesh->set_instance_custom_data(i,it.value.custom_data);
        mesh_id_maps[it.key] = i;
        ++i;
    }
}
void SceneChunk::MeshInstance::set_mesh_transform(int mesh_id,const Transform3D& t) {
    auto it = mesh_transforms.find(mesh_id);
    if (it != mesh_transforms.end()) {
        it->value.transform = t;
        dirty = true;
    }
    else {
        mesh_transforms[mesh_id] = MeshInstanceInfo();
        mesh_transforms[mesh_id].transform = t;
        dirty = true;
    }
}
void SceneChunk::MeshInstance::set_mesh_color(int mesh_id,const Color& color) {
    auto it = mesh_transforms.find(mesh_id);
    if (it != mesh_transforms.end()) {
        it->value.color = color;
        dirty = true;
    }
}
void SceneChunk::MeshInstance::set_mesh_custom_data(int mesh_id,const Color& color) {
    auto it = mesh_transforms.find(mesh_id);
    if (it != mesh_transforms.end()) {
        it->value.custom_data = color;
        dirty = true;
    }
}

/*********************************************************************************************************/

int SceneChunkGroupInstance::_add_mesh_instance(const String& p_path,const Transform3D& t) {
    
    SceneChunk* chunk = Object::cast_to<SceneChunk>(ObjectDB::get_instance(chunk_id));
    if (chunk) {
        return chunk->add_multimesh_instance(p_path, t);
    }
    return -1;

}
int SceneChunkGroupInstance::_add_collision_instance(const Transform3D& t,SceneDataCompoent::CollisionShapeType type,const Vector3& box_size,float height,float radius) {
    
    SceneChunk* chunk = Object::cast_to<SceneChunk>(ObjectDB::get_instance(chunk_id));
    if (chunk) {
        return chunk->add_collision_instance(t,type,box_size,height,radius);
    }
    return -1;

}
int SceneChunkGroupInstance::_add_mesh_collision_instance(const String& p_path,const Transform3D& t) {
    
    SceneChunk* chunk = Object::cast_to<SceneChunk>(ObjectDB::get_instance(chunk_id));
    if (chunk) {
        return chunk->add_collision_instance(t,SceneDataCompoent::CollisionShapeType::SHAPE_MESH,Vector3(),0,0);
    }
    return -1;

}

void SceneChunkGroupInstance::set_lod(int p_lod) {
    if(curr_lod == p_lod) {
        return;
    }
    curr_lod = p_lod;
    clear_show_instance_ids();
    SceneChunk* chunk = Object::cast_to<SceneChunk>(ObjectDB::get_instance(chunk_id));
    if (chunk) {
        if(resource.is_null()) {
            return;
        }
        Transform3D trans = get_global_transform();
        resource->show(curr_lod,p_lod,trans,this);
    }
    
}
SceneChunk* SceneChunkGroupInstance::get_chunk() {
    return Object::cast_to<SceneChunk>(ObjectDB::get_instance(chunk_id));
}
void SceneChunkGroupInstance::clear_show_instance_ids() {
    SceneChunk* chunk = Object::cast_to<SceneChunk>(ObjectDB::get_instance(chunk_id));
    if (chunk) {
        for(auto& it : curr_show_meshinstance_ids) {
            chunk->remove_multimesh_instance(it.value, it.key);
        }
        for(auto& it : curr_show_collision_ids) {
            chunk->remove_collision_instance( it);
        }
        for(auto& it : curr_show_mesh_collision_ids) {
            chunk->remove_mesh_collision_instance(it.key, it.value);
        }
    }
}

/*********************************************************************************************************/

int SceneChunk::add_multimesh_instance(const String& res_path, const Transform3D& t) {

    Ref<MeshInstance> mesh_instance;
    if(!mult_mesh_instances.has(res_path)) {
        Ref<ResourceLoader::LoadToken> token = ResourceLoader::_load_start(res_path,"",ResourceLoader::LOAD_THREAD_FROM_CURRENT, ResourceFormatLoader::CACHE_MODE_IGNORE);
        if(token.is_null()) {
            return -1;
        }
        mesh_instance = Ref<MeshInstance>(memnew(MeshInstance));
        mesh_instance->load_token = token;
        mult_mesh_instances[res_path] = mesh_instance;
    }
    else {
        mesh_instance = mult_mesh_instances[res_path];
    }
    int id = get_free_id();

    mesh_instance->set_mesh_transform(id,t);
    return id;
}
void SceneChunk::remove_multimesh_instance(const String& res_path, int id) {
    if(mult_mesh_instances.has(res_path)) {
        Ref<MeshInstance> mesh_instance = mult_mesh_instances[res_path];
        mesh_instance->remove_instance(id);
        unuse_id_list.push_back(id);
    }
}

int SceneChunk::add_collision_instance(const Transform3D& t,SceneDataCompoent::CollisionShapeType type,const Vector3& box_size,float height,float radius) {
    int id = -1;
    switch (type)
    {
    case SceneDataCompoent::CollisionShapeType::SHAPE_MESH:
        break;
    case SceneDataCompoent::CollisionShapeType::SHAPE_BOX:
    {
        id = get_free_id();
        RID box_shape = PhysicsServer3D::get_singleton()->box_shape_create();
        PhysicsServer3D::get_singleton()->shape_set_data(box_shape,box_size);
		RID owner_id = PhysicsServer3D::get_singleton()->body_create();

        Collision& collision = collision_instances[id];
        collision.node_id = owner_id;
        collision.shape = box_shape;
        collision.collision_layer = 1;
        collision.collision_mask = 1;
        PhysicsServer3D::get_singleton()->body_set_collision_layer(owner_id, collision.collision_layer);
        PhysicsServer3D::get_singleton()->body_set_collision_mask(owner_id, collision.collision_mask);
        PhysicsServer3D::get_singleton()->body_add_shape(owner_id, box_shape, t);
		PhysicsServer3D::get_singleton()->body_set_mode(owner_id, PhysicsServer3D::BODY_MODE_STATIC);
    }
        break;
    case SceneDataCompoent::CollisionShapeType::SHAPE_CAPSULE:
    {
        id = get_free_id();
        RID capsule_shape = PhysicsServer3D::get_singleton()->capsule_shape_create();
        Dictionary new_d;
        new_d["radius"] = radius;
        new_d["height"] = height;
        PhysicsServer3D::get_singleton()->shape_set_data(capsule_shape,new_d);
		RID owner_id = PhysicsServer3D::get_singleton()->body_create();

        Collision& collision = collision_instances[id];
        collision.node_id = owner_id;
        collision.shape = capsule_shape;
        collision.collision_layer = 1;
        collision.collision_mask = 1;
        PhysicsServer3D::get_singleton()->body_set_collision_layer(owner_id, collision.collision_layer);
        PhysicsServer3D::get_singleton()->body_set_collision_mask(owner_id, collision.collision_mask);
        PhysicsServer3D::get_singleton()->body_add_shape(owner_id, capsule_shape, t);
		PhysicsServer3D::get_singleton()->body_set_mode(owner_id, PhysicsServer3D::BODY_MODE_STATIC);
    }
        break;
    case SceneDataCompoent::CollisionShapeType::SHAPE_CYLINDER:
    {
        id = get_free_id();
        RID cylinder_shape = PhysicsServer3D::get_singleton()->cylinder_shape_create();
		Dictionary new_d;
        new_d["radius"] = radius;
        new_d["height"] = height;
        PhysicsServer3D::get_singleton()->shape_set_data(cylinder_shape,new_d);
		RID owner_id = PhysicsServer3D::get_singleton()->body_create();

        Collision& collision = collision_instances[id];
        collision.node_id = owner_id;
        collision.shape = cylinder_shape;
        collision.collision_layer = 1;
        collision.collision_mask = 1;
        PhysicsServer3D::get_singleton()->body_set_collision_layer(owner_id, collision.collision_layer);
        PhysicsServer3D::get_singleton()->body_set_collision_mask(owner_id, collision.collision_mask);
        PhysicsServer3D::get_singleton()->body_add_shape(owner_id, cylinder_shape, t);
		PhysicsServer3D::get_singleton()->body_set_mode(owner_id, PhysicsServer3D::BODY_MODE_STATIC);
    }
        break;
    case SceneDataCompoent::CollisionShapeType::SHAPE_SPHERE:
    {
        id = get_free_id();
        RID sphere_shape = PhysicsServer3D::get_singleton()->sphere_shape_create();
        PhysicsServer3D::get_singleton()->shape_set_data(sphere_shape,radius);
	    RID owner_id = PhysicsServer3D::get_singleton()->body_create();

        Collision& collision = collision_instances[id];
        collision.node_id = owner_id;
        collision.shape = sphere_shape;
        collision.collision_layer = 1;
        collision.collision_mask = 1;
        PhysicsServer3D::get_singleton()->body_set_collision_layer(owner_id, collision.collision_layer);
        PhysicsServer3D::get_singleton()->body_set_collision_mask(owner_id, collision.collision_mask);
        PhysicsServer3D::get_singleton()->body_add_shape(owner_id, sphere_shape, t);
		PhysicsServer3D::get_singleton()->body_set_mode(owner_id, PhysicsServer3D::BODY_MODE_STATIC);
    }
        break;
    default:
        break;
    }
    return id;
}
void SceneChunk::remove_collision_instance( int id) {
    if(collision_instances.has(id)) {
        Collision& collision = collision_instances[id];
        PhysicsServer3D::get_singleton()->free(collision.shape);
        PhysicsServer3D::get_singleton()->free(collision.node_id);
		collision_instances.erase(id);
    }
}


int SceneChunk::add_mesh_collision_instance(const Transform3D& t,const String& p_path) {
    Ref<MeshCollisionInstance> mesh_collision_instance;
    if(mesh_collision_instances.has(p_path)) {
        mesh_collision_instance = mesh_collision_instances[p_path];
    }
    else {
        Ref<MeshCollisionResource> mesh_collision_resource = ResourceLoader::load(p_path);
        if(mesh_collision_resource.is_null()) {
            return -1;
        }
        mesh_collision_instance.instantiate();
        mesh_collision_instances[p_path] = mesh_collision_instance;
        RID shape = PhysicsServer3D::get_singleton()->concave_polygon_shape_create();
	    PhysicsServer3D::get_singleton()->shape_set_data(shape, mesh_collision_resource->get_points());
		mesh_collision_instance->shape = shape;
    }
	RID owner_id = PhysicsServer3D::get_singleton()->body_create();

	int id = get_free_id();
    Collision& collision = mesh_collision_instance->mesh_transforms[id];
    PhysicsServer3D::get_singleton()->body_add_shape(owner_id, mesh_collision_instance->shape, t);
    PhysicsServer3D::get_singleton()->body_set_mode(owner_id, PhysicsServer3D::BODY_MODE_STATIC);
    PhysicsServer3D::get_singleton()->body_set_collision_layer(owner_id, collision.collision_layer);
    PhysicsServer3D::get_singleton()->body_set_collision_mask(owner_id, collision.collision_mask);
	return id;
}
void SceneChunk::remove_mesh_collision_instance( int id,const String& p_path) {
    if(mesh_collision_instances.has(p_path)) {
        Ref<MeshCollisionInstance> mesh_collision_resource = mesh_collision_instances[p_path];
        if(mesh_collision_resource.is_null()) {
            return;
        }
        if(mesh_collision_resource->mesh_transforms.has(id)) {            
            Collision& collision = mesh_collision_resource->mesh_transforms[id];
            PhysicsServer3D::get_singleton()->free(collision.node_id);
            mesh_collision_resource->mesh_transforms.erase(id);
        }
    }   
}
