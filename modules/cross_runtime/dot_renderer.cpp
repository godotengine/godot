/*
Actual renderer implementation:
- creates a quad mesh
- creates many instances of it
- reads positions from wasm memory
- updates instance transforms
- draws them
*/
#include "dot_renderer.h" //imports the class DotRenderer
#include "memory_layout.h" //the memory contract

#include "core/math/vector2.h" //for positions and scaling
#include "scene/resources/mesh.h" // for ArrayMesh

void DotRenderer::_bind_methods() {}

DotRenderer::DotRenderer() {}

//event handler - Godot sends events, This reacts depending on type
void DotRenderer::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_READY: { 
            //print_line("[DotRenderer] NOTIFICATION_READY");
            set_process(true); //enables processing
            _setup_multimesh(); //initializes the multimesh
        } break;

        case NOTIFICATION_PROCESS: {
            _update_multimesh(); //update transforms from entity memory
            queue_redraw(); //request redraws
        } break;

        case NOTIFICATION_DRAW: { //godot asks node to draw
            _draw_multimesh(); 
        } break;
    }
}

//initialization of multimesh
void DotRenderer::_setup_multimesh() {
    // builds a quad multimesh
    Ref<ArrayMesh> array_mesh;
    array_mesh.instantiate(); //allocates actual object

    PackedVector2Array vertices; //array of 2D positions
    vertices.push_back(Vector2(-1, -1));   // bottom-left
    vertices.push_back(Vector2( 1, -1));   // bottom-right
    vertices.push_back(Vector2( 1,  1));   // top-right
    vertices.push_back(Vector2(-1,  1));   // top-left

    PackedInt32Array indices; //defines triangles
    indices.push_back(0); indices.push_back(1); indices.push_back(2);
    indices.push_back(0); indices.push_back(2); indices.push_back(3);

    Array surface_array; //mesh data container
    surface_array.resize(Mesh::ARRAY_MAX); //prepares all slots
    //helps mesh know the geometry
    surface_array[Mesh::ARRAY_VERTEX] = vertices;
    surface_array[Mesh::ARRAY_INDEX]  = indices;
    //creates the mesh surface
    array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_array);

    // creates multimesh oject
    multimesh.instantiate();
    multimesh->set_mesh(array_mesh); //each instance uses this quad
    multimesh->set_transform_format(MultiMesh::TRANSFORM_2D); //each instance has 2D node transform
    multimesh->set_use_colors(false); //no colors
    multimesh->set_instance_count(ENTITY_COUNT); //creates one instance per entity count

    // 3. Place all instances far off-screen until the worker is ready
    Transform2D off_screen; //transforms the object
    off_screen.set_origin(Vector2(-10000, -10000)); //moves far away

    //apply to all instances
    for (int i = 0; i < ENTITY_COUNT; ++i) {
        multimesh->set_instance_transform_2d(i, off_screen);
    }

    //this sets off updates
    multimesh_initialized = true;
}

//runs every frame
void DotRenderer::_update_multimesh() {
    //avoids updating before setup
    if (!multimesh_initialized) {
        static bool once = false;
        if (!once) {
            once = true;
        }
        return;
    }
    //pointer to worker ready offset - it is always reread
    const volatile int32_t *worker_ready =
        reinterpret_cast<const volatile int32_t *>(WORKER_READY_OFFSET);

    //logs changes
    static int last_ready = -1;
    if (*worker_ready != last_ready) { 
        last_ready = *worker_ready;
    }

    if (*worker_ready == 0) { //if worker not ready, don't render entities to prevent reading invalid or partial data
        return;
    }

    const Entity *e = reinterpret_cast<const Entity *>(ENTITIES_OFFSET); //points t entity array at entity offset
    const float dot_size_local = dot_size; //avoids repeated membber access

    

    for (std::uint32_t i = 0; i < ENTITY_COUNT; ++i) { //for each entity
        //read x,y - it reads directly from memory
        float x = e[i].x; 
        float y = e[i].y;

        //clamp logic
        if (x < 0) x = -x; else if (x > screen_width) x = 2.0f * screen_width - x;
        if (y < 0) y = -y; else if (y > screen_height) y = 2.0f * screen_height - y;

        //per instance transform
        Transform2D t;
        t.set_origin(Vector2(x, y)); //move dot
        t.set_scale(Vector2(dot_size_local, dot_size_local)); //scale quad to dot size

        multimesh->set_instance_transform_2d(i, t); //updates transform
    }

}

//actual drawing
void DotRenderer::_draw_multimesh() {
    if (multimesh.is_valid()) { //avoids null access
        draw_multimesh(multimesh, Ref<Texture2D>()); //renders all instances
    } 
}
