#ifndef RENDERING_SERVER_3DS_H
#define RENDERING_SERVER_3DS_H

class RenderingServer3DS {
public:
    RenderingServer3DS();
    ~RenderingServer3DS();

    void initialize();
    void draw_frame();
};

#endif
