/* Suppress the warnings for this include, since we don't care about them for external dependencies
 * Requires at least GCC 4.6 or higher
*/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
#include "../ext_deps/miniz.c"
#include "../ext_deps/mjson.h"
#pragma GCC diagnostic pop

//Temporary, will get 'structed up'
static double acc_bias[3];
static double acc_scale[3];
static double gyro_bias[3];
static double gyro_scale[3];
//end of temporary

//Vive config packet
const struct json_attr_t sensor_offsets[] = {
    {"acc_bias", t_array,   .addr.array.element_type = t_real,
                            .addr.array.arr.reals.store = acc_bias,
                            .addr.array.maxlen = 3},
    {"acc_scale", t_array,  .addr.array.element_type = t_real,
                            .addr.array.arr.reals.store = acc_scale,
                            .addr.array.maxlen = 3},
    {"device", t_object},
    {"device_class", t_ignore},
    {"device_pid", t_ignore},
    {"device_serial_number", t_ignore},
    {"device_vid", t_ignore},
    {"display_edid", t_ignore},
    {"display_gc", t_ignore},
    {"display_mc", t_ignore},
    {"gyro_bias", t_array,  .addr.array.element_type = t_real,
                            .addr.array.arr.reals.store = gyro_bias,
                            .addr.array.maxlen = 3},
    {"gyro_scale", t_array, .addr.array.element_type = t_real,
                            .addr.array.arr.reals.store = gyro_scale,
                            .addr.array.maxlen = 3},
    {NULL},
};