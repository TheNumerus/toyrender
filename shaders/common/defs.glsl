#define GLOBAL uniform Global { \
    float exposure; \
    int debug; \
    float res_x; \
    float res_y; \
    float time; \
    int frame_index; \
    int half_res; \
} globals

#define VIEW_PROJ uniform UniformBufferObject { \
    mat4 view[2]; \
    mat4 proj[2]; \
} view_proj