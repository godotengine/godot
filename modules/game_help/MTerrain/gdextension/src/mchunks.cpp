#include "mchunks.h"

#include "mchunk_generator.h"
#include "mconfig.h"

#include "core/variant/variant.h"

#ifdef M_DEBUG
#include "core/io/resource_loader.h"
#endif

int MChunks::number_of_user = 0;
Vector<Ref<Mesh>> MChunks::meshes;
HashMap<int64_t,int> MChunks::mesh_hash;

void MChunks::remove_user(){
    number_of_user--;
    if(MChunks::number_of_user==0){
        meshes.clear();
        MChunks::mesh_hash.clear();
    }
}

void MChunks::_bind_methods() {
    ClassDB::bind_method(D_METHOD("create_chunks","_min_size","_max_size","_min_h_scale","_max_h_scale", "size_info"), &MChunks::create_chunks);
}

RID MChunks::get_mesh(int32_t size_meter, real_t h_scale, int8_t edge,const Ref<Material>& _material){
    String skey = itos(size_meter) + "_" + rtos(h_scale) + "_" + itos(edge);
    int64_t key = skey.hash();
    if(mesh_hash.has(key)){
        int index = mesh_hash[key];
        return meshes[index]->get_rid();
    }
    Ref<Mesh> mesh;
    if(edge == M_MAIN){
        mesh = MChunkGenerator::generate(size_meter,h_scale,false,false,false,false);
    } else if(edge == M_L){
        mesh = MChunkGenerator::generate(size_meter,h_scale,true,false,false,false);
    }else if (edge == M_R)
    {
        mesh = MChunkGenerator::generate(size_meter,h_scale,false,true,false,false);
    }else if (edge == M_T)
    {
        mesh = MChunkGenerator::generate(size_meter,h_scale,false,false,true,false);
    }else if (edge == M_B)
    {
        mesh = MChunkGenerator::generate(size_meter,h_scale,false,false,false,true);
    }else if (edge == M_LT)
    {
        mesh = MChunkGenerator::generate(size_meter,h_scale,true,false,true,false);
    }else if (edge == M_RT)
    {
        mesh = MChunkGenerator::generate(size_meter,h_scale,false,true,true,false);
    }else if (edge == M_LB)
    {
        mesh = MChunkGenerator::generate(size_meter,h_scale,true,false,false,true);
    }else if (edge == M_RB)
    {
        mesh = MChunkGenerator::generate(size_meter,h_scale,false,true,false,true);
    }else if (edge == M_LRTB)
    {
        mesh = MChunkGenerator::generate(size_meter,h_scale,true,true,true,true);
    }
    if(_material.is_valid()){
        mesh->surface_set_material(0,_material);
    }
    meshes.append(mesh);
    int index = meshes.size() - 1;
    mesh_hash.insert(key, index);
    return mesh->get_rid();
}

MChunks::MChunks(){
    MChunks::number_of_user++;
    #ifdef M_DEBUG
    for(int i=0; i < 6; i++){
        Ref<Material> m = ResourceLoader::load("res://addons/m_terrain/debug_mat/lod"+itos(i)+".material");
        debug_material.append(m);
    }
    #endif
}

MChunks::~MChunks(){
    MChunks::remove_user();
}

void MChunks::create_chunks(int32_t _min_size, int32_t _max_size, real_t _min_h_scale, real_t _max_h_scale, Array info) {
    base_size_meter = _min_size;
    h_scale = _min_h_scale;
    int8_t size = 0;
    for(int32_t size_meter=_min_size;size_meter<=_max_size;size_meter*=2){
        int8_t lod = 0;
        Array size_info = info[size];
        MSize current_size;
        for(real_t hs=_min_h_scale; hs<=_max_h_scale; hs*=2){
            MLod current_lod;
            if(size_info[lod]){
                int8_t max_edge = (hs==_max_h_scale) ? 1 : M_MAX_EDGE;
                for(int8_t edge=0; edge<max_edge;edge++){
                    Ref<Material> mat;
                    #ifdef M_DEBUG
                    mat = (lod<6) ? debug_material[lod] : nullptr;
                    #endif
                    current_lod.meshes.append(get_mesh(size_meter,hs,edge, mat));
                }
            }
            current_size.lods.append(current_lod);
            lod++;
        }
        sizes.append(current_size);
        size++;
    }
    max_size = log2(_max_size/_min_size);
    max_lod = log2(_max_h_scale/_min_h_scale);
}
