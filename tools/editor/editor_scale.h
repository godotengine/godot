#ifndef EDITOR_SCALE_H
#define EDITOR_SCALE_H

void editor_set_scale(float p_scale);
float editor_get_scale();

#define EDSCALE (editor_get_scale())
#endif // EDITOR_SCALE_H
