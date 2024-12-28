#include "scene_chunk.h"




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
