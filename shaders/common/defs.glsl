#define GLOBAL uniform Global { \
    float exposure; \
    int debug; \
    float res_x; \
    float res_y; \
    float time; \
    int frame_index; \
    int half_res; \
    vec2 curr_jitter; \
    vec2 prev_jitter; \
} globals

#define VIEW_PROJ uniform UniformBufferObject { \
    mat4 view[2]; \
    mat4 proj[2]; \
} view_proj

#define ENV uniform Environment { \
    vec3 sun_direction; \
    vec3 sun_color; \
} env