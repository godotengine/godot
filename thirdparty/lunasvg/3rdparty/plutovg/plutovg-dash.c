#include "plutovg-private.h"

#include <math.h>

plutovg_dash_t* plutovg_dash_create(double offset, const double* data, int size)
{
    if(data==NULL || size==0)
        return NULL;

    plutovg_dash_t* dash = malloc(sizeof(plutovg_dash_t));
    dash->offset = offset;
    dash->data = malloc((size_t)size * sizeof(double));
    dash->size = size;
    memcpy(dash->data, data, (size_t)size * sizeof(double));
    return dash;
}

plutovg_dash_t* plutovg_dash_clone(const plutovg_dash_t* dash)
{
    if(dash==NULL)
        return NULL;

    return plutovg_dash_create(dash->offset, dash->data, dash->size);
}

void plutovg_dash_destroy(plutovg_dash_t* dash)
{
    if(dash==NULL)
        return;

    free(dash->data);
    free(dash);
}

plutovg_path_t* plutovg_dash_path(const plutovg_dash_t* dash, const plutovg_path_t* path)
{
    if(dash->data==NULL || dash->size==0)
        return plutovg_path_clone(path);

    int toggle = 1;
    int offset = 0;
    double phase = dash->offset;
    while(phase >= dash->data[offset])
    {
        toggle = !toggle;
        phase -= dash->data[offset];
        offset += 1;
        if(offset == dash->size) offset = 0;
    }

    plutovg_path_t* flat = plutovg_path_clone_flat(path);
    plutovg_path_t* result = plutovg_path_create();
    plutovg_array_ensure(result->elements, flat->elements.size);
    plutovg_array_ensure(result->points, flat->points.size);

    plutovg_path_element_t* elements = flat->elements.data;
    plutovg_path_element_t* end = elements + flat->elements.size;
    plutovg_point_t* points = flat->points.data;
    while(elements < end)
    {
        int itoggle = toggle;
        int ioffset = offset;
        double iphase = phase;

        double x0 = points->x;
        double y0 = points->y;

        if(itoggle)
            plutovg_path_move_to(result, x0, y0);

        ++elements;
        ++points;

        while(elements < end && *elements==plutovg_path_element_line_to)
        {
            double dx = points->x - x0;
            double dy = points->y - y0;
            double dist0 = sqrt(dx*dx + dy*dy);
            double dist1 = 0;

            while(dist0 - dist1 > dash->data[ioffset] - iphase)
            {
                dist1 += dash->data[ioffset] - iphase;
                double a = dist1 / dist0;
                double x = x0 + a * dx;
                double y = y0 + a * dy;

                if(itoggle)
                    plutovg_path_line_to(result, x, y);
                else
                    plutovg_path_move_to(result, x, y);

                itoggle = !itoggle;
                iphase = 0;
                ioffset += 1;
                if(ioffset == dash->size) ioffset = 0;
            }

            iphase += dist0 - dist1;

            x0 = points->x;
            y0 = points->y;

            if(itoggle)
                plutovg_path_line_to(result, x0, y0);

            ++elements;
            ++points;
        }
    }

    plutovg_path_destroy(flat);
    return result;
}
