#pragma once

#include "scene/2d/node_2d.h" //imports Node2D for 2D scene objects
#include "core/templates/vector.h" //Godots internal dynamic array
#include "core/math/transform_2d.h" //2D transform math
#include "scene/resources/multimesh.h" //Multimesh


//Class DotRenderer which inherits from Node2D - ensures it gains all Node2D behaviour
class DotRenderer : public Node2D {
    GDCLASS(DotRenderer, Node2D); // hooks the class for registration

protected:
    static void _bind_methods(); 

public:
    DotRenderer();

    void _notification(int p_what); //tells which event  occured  ready, process, draw

private:
    void _setup_multimesh(); //creates multimesh
    void _update_multimesh(); //updates by reading then updating
    void _draw_multimesh(); //draws

    Ref<MultiMesh> multimesh; //Godot's reference counted smart pointer
    bool multimesh_initialized = false; //tracks multimesh initialization

    float screen_width = 1024.0f;
    float screen_height = 768.0f;
    float dot_size = 2.0f;
};
