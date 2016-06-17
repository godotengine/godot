#ifndef EDITOR_SCALE_H
#define EDITOR_SCALE_H

void editor_set_hidpi(bool p_hidpi);
bool editor_is_hidpi();

#define EDSCALE (editor_is_hidpi() ? 2 : 1)
#endif // EDITOR_SCALE_H
