import Common;
import Sampling;
import Sky;

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_shader_realtime_clock : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define PI 3.141569

struct Payload {
    bool isMiss;
    float dist;
    float3 normal;
    float3 color;
    float3 hit;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(1, 0)]]
RaytracingAccelerationStructure tlas;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(3, 0)]]
ConstantBuffer<Environment> env;

[[vk::binding(5, 0)]]
StructuredBuffer<MeshInstanceInfo> meshData;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int sample;
    int bounces;
    int out_idx;
    int pdf_idx;
    int importance_idx;
    int use_importance;
    int sky_idx;
    float direct_trace_distance;
    float indirect_trace_distance;
    float fov;
    float indirect_intensity_clamp;
};

[vk::push_constant]
uniform PushConsts pc;

[vk::constant_id(0)]
const SkyMode skyVariant;

static uint hash_state;

float pcg_hash() {
    uint state = hash_state;
    hash_state = hash_state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint res = (word >> 22u) ^ word;
    return float(res % 4096) / 4096.0;
}

void compute_default_basis(const float3 normal, out float3 x, out float3 y) {
    float3 z  = normal;
    const float yz = -z.y * z.z;
    y = normalize(((abs(z.z) > 0.99999f) ? float3(-z.x * z.y, 1.0f - z.y * z.y, yz) : float3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

    x = cross(y, z);
}

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

void tracePrimary(float3 pos, float3 dir, inout Payload payload) {
    RayDesc ray;
    ray.Direction = dir;
    ray.Origin = pos;
    ray.TMax = 1000.0;
    ray.TMin = 0.0001;

    TraceRay(
        tlas,
        RAY_FLAG_FORCE_OPAQUE,
        0xFF,
        0,
        0,
        0,
        ray,
        payload
    );
}


void trace(float3 pos, float3 dir, inout Payload payload) {
    RayDesc ray;
    ray.Direction = dir;
    ray.Origin = pos;
    ray.TMax = pc.indirect_trace_distance;
    ray.TMin = 0.0001;

    TraceRay(
        tlas,
        RAY_FLAG_FORCE_OPAQUE,
        0xFF,
        0,
        0,
        0,
        ray,
        payload
    );
}

void traceShadow(float3 pos, float3 dir, inout Payload payload) {
    RayDesc ray;
    ray.Direction = dir;
    ray.Origin = pos;
    ray.TMax = pc.direct_trace_distance;
    ray.TMin = 0.0001;

    payload.isMiss = false;
    TraceRay(
        tlas,
        RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xFF,
        0,
        0,
        1,
        ray,
        payload
    );
}

float signNotZero(in float k) {
    return (k >= 0.f) ? 1.f : -1.f;
}

float2 signNotZero(in float2 v) {
    return float2(signNotZero(v.x), signNotZero(v.y));
}

float2 octEncode(float3 v) {
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    float2 result = v.xy * (1.f / l1norm);
    if (v.z < 0.f) {
        result = (1.f - abs(result.yx)) * signNotZero(result.xy);
    }
    return result;
}

float3 octDecode(float2 o) {
    float3 v = float3(o.x, o.y, 1.f - abs(o.x) - abs(o.y));
    if (v.z < 0.f) {
        v.xy = (1.f - abs(v.yx)) * signNotZero(v.xy);
    }
    return normalize(v);
}

uint time_diff(uint startTime, uint endTime) {
    return endTime >= startTime ? (endTime-startTime) : (~0u-(startTime-endTime));
}

float3 generatePinholeRay(int2 pixel, int width, int height) {
    float tanHalfAngle = tan(pc.fov / 2.f);
    return normalize(float3(float2(2 * (pixel.x + global.curr_jitter.x) - width, -2 * (pixel.y + global.curr_jitter.y) + height) * tanHalfAngle / width, -1));
}

constexpr static float origin = 1.0f / 32.0f;
constexpr static float float_scale = 1.0f / 65536.0f;
constexpr static float int_scale = 256.0f;

float3 offset_ray(const float3 p, const float3 n) {
    int3 of_i = int3(int(int_scale * n.x), int(int_scale * n.y), int(int_scale * n.z));

    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z))
    );

    return float3(
        abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z
    );
}

float3 lightSample(float3 pos, float3 normal, float3 dir_ray, float3 color, in Payload payload) {
    float3 radiance = 0.0;

    // select light
    float sun_light_intensity = max(dot(dir_ray, normal) * env.sun_color_intensity.w, 0.0) * luminance(env.sun_color_intensity.xyz);
    float sum = sun_light_intensity;

    float2 rand_vars_sphere = float2(pcg_hash(), pcg_hash());

    for (int i = 0; i < env.point_lights; i++) {
        let light = env.lights[i];
        let light_pos = light.pos_radius.xyz + rand_sphere(rand_vars_sphere) * light.pos_radius.w;
        let dir = normalize(light_pos - pos);
        let len = distance(light_pos, pos);
        let intensity = ((light.color_intensity.w * luminance(light.color_intensity.xyz)) / (len * len));
        let angle = max(dot(normal, dir), 0.0);
        sum += intensity * angle;
    }

    // add sky
    float3 hemi = rand_hemi(rand_vars_sphere);

    float sky_intensity;
    if (pc.use_importance == 1 && skyVariant == SkyMode::Textured) {
        float2 mapped = rand_vars_sphere;
        mapped.y = storages[pc.importance_idx].Load(int2(1024, int(floor(rand_vars_sphere.y * 1024)))).x;
        mapped.x = storages[pc.importance_idx].Load(int2(int(floor(rand_vars_sphere.x * 1024)), int(floor(mapped.y * 1024)))).x;

        float3 dir = octDecode((mapped - 0.5) * 2.0);
        if (dot(normal, dir) >= 0.0) {
            float3 sky_color = Toyrender::Sky::sample(dir, images[pc.sky_idx], env.rotation);

            float pdf = images[pc.pdf_idx].SampleLevelZero(mapped).x;
            const float uniformPdf = 1.0 / (1024.0 * 1024.0);
            // surface of hemi is 2*pi*r^2, circle is pi*r^2 -> 2x factor + additional 2x factor because of dot -- half of samples can go below surface
            float adjust = 4.0 * uniformPdf / (pdf);

            sky_intensity = env.sky_color_intensity.w * adjust * dot(normal, dir) * luminance(sky_color);
        }
    } else {
        float3 tangent, bitangent;
        compute_default_basis(normal, tangent, bitangent);
        let dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * normal;

        float3 sky_color;

        switch (skyVariant) {
        case SkyMode::Shader:
            sky_color = Toyrender::Sky::sample_distorted(dir, images[pc.sky_idx]);
            break;
        case SkyMode::SingleColor:
            sky_color = env.sky_color_intensity.xyz;
            break;
        case SkyMode::Textured:
            sky_color = Toyrender::Sky::sample(dir, images[pc.sky_idx], env.rotation);
            break;
        }

        sky_intensity = env.sky_color_intensity.w * luminance(sky_color);
    }
    sum += sky_intensity;


    float rand_light = pcg_hash() * sum;

    if (rand_light > (sun_light_intensity + sky_intensity)) {
        sum = sun_light_intensity;
        var light : Optional<PointLight> = none;
        float3 light_pos;
        float3 dir;
        float len;
        float angle;
        float intensity;

        for (int i = 0; i < env.point_lights; i++) {
            light = env.lights[i];
            light_pos = light.value.pos_radius.xyz + rand_sphere(rand_vars_sphere) * light.value.pos_radius.w;
            dir = normalize(light_pos - pos);
            len = distance(light_pos, pos);
            intensity = ((light.value.color_intensity.w * luminance(light.value.color_intensity.xyz)) / (len * len));
            angle = max(dot(normal, dir), 0.0);
            sum += intensity * angle;
            if (sum > rand_light) {
                break;
            }
        }

        RayDesc ray;
        ray.Direction = dir;
        ray.Origin = pos;
        ray.TMax = len;
        ray.TMin = 0.0001;

        payload.isMiss = false;
        TraceRay(
            tlas,
            RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            0xFF,
            0,
            0,
            1,
            ray,
            payload
        );

        float compensate = max(sum / max(intensity * angle, 0.000001), 1.0);

        float shadow = float(payload.isMiss) * max(dot(ray.Direction, normal), 0.0);
        radiance = shadow * intensity * light.value.color_intensity.xyz * color * compensate;
    } else if (rand_light > (sun_light_intensity)) {
        if (pc.use_importance == 1 && skyVariant == SkyMode::Textured) {
            float2 mapped = rand_vars_sphere;
            mapped.y = storages[pc.importance_idx].Load(int2(1024, int(floor(rand_vars_sphere.y * 1024)))).x;
            mapped.x = storages[pc.importance_idx].Load(int2(int(floor(rand_vars_sphere.x * 1024)), int(floor(mapped.y * 1024)))).x;

            float3 dir = octDecode((mapped - 0.5) * 2.0);
            if (dot(normal, dir) >= 0.0) {
                trace(offset_ray(pos, normal), dir, payload);

                if (payload.isMiss) {
                    float pdf = images[pc.pdf_idx].SampleLevelZero(mapped).x;
                    float uniformPdf = 1.0 / (1024.0 * 1024.0);
                    // surface of hemi is 2*pi*r^2, circle is pi*r^2 -> 2x factor + additional 2x factor because of dot -- half of samples can go below surface
                    float adjust = 4.0 * uniformPdf / (pdf);

                    radiance += env.sky_color_intensity.w * color * adjust * dot(normal, dir) * payload.color;
                }
            }
        } else {
            float3 tangent, bitangent;
            compute_default_basis(normal, tangent, bitangent);
            let dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * normal;

            trace(offset_ray(pos, normal), dir, payload);
            radiance += env.sky_color_intensity.w * color * float(payload.isMiss) * payload.color;
        }
    } else {
        float compensate = max(sum / max(sun_light_intensity, 0.000001), 1.0);
        if (dot(dir_ray, normal) > 0.0 && env.sun_color_intensity.w > 0.0) {
            payload.isMiss = true;
            traceShadow(pos, dir_ray, payload);
        }

        float shadow = float(payload.isMiss) * max(dot(dir_ray, normal), 0.0);
        radiance = shadow * env.sun_color_intensity.xyz * env.sun_color_intensity.w * color * compensate;
    }

    return radiance;
}


[shader("raygeneration")]
void raygen() {
    uint3 launch_idx = DispatchRaysIndex();

    int2 size = int2(int(global.res_x), int(global.res_y));

    if (launch_idx.x >= size.x || launch_idx.y >= size.y) {
        return;
    }

    Payload payload;

    payload.isMiss = false;
    payload.color = float3(0.0);

    uint start = getRealtimeClockLow().x;

    float3 indirect = float3(0.0);
    float3 direct = float3(0.0);

    float3 tangent, bitangent;

    // setup hasher with min index 1 so that even first sample is noisy
    hash_state = (launch_idx.x + launch_idx.y * uint(global.res_x)) * (global.frame_index + 1);

    float2 rand = float2(pcg_hash(), pcg_hash());

    compute_default_basis(env.sun_dir_sun_angle.xyz, tangent, bitangent);

    float3 cone = rand_cone(rand, cos(env.sun_dir_sun_angle.w));
    float3 ray_shadow_dir = cone.x * tangent + cone.y * bitangent + cone.z * env.sun_dir_sun_angle.xyz;

    // scale of energy for subsequent bounces
    float3 intensity_indirect = 1.0;

    float3 dir = generatePinholeRay(int2(launch_idx.x, launch_idx.y), int(global.draw_res_x), int(global.draw_res_y));

    dir = mul(float3x3(vp.view_inverse), dir).xyz;

    float3 pos = transpose(vp.view_inverse)[3].xyz;
    float3 hitNormal;

    float3 primaryPos; 
    float3 primaryNormal;

    tracePrimary(pos, dir, payload);

    float3 primaryColor = payload.color;

    pos = payload.hit;

    if (payload.isMiss) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                payload.color * env.sky_color_intensity.w,
                0.0
            )
        );
        return;
    } else {
        primaryPos = pos;

        hitNormal = payload.normal;
        primaryNormal = hitNormal;
    }

    if (global.debug == DebugMode::Normal) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                (primaryNormal * 0.5) + 0.5,
                0.0
            )
        );
        return;
    }

    // TODO model proper energy passing from material
    float bounce_energy = 0.95;

    float roughness = 0.3;

    // current surface_color
    float3 current_color = payload.color;

    for (int i = 0; i < pc.bounces; i++) {
        compute_default_basis(hitNormal, tangent, bitangent);

        float d = roughness + (1.0 - roughness) * pow(1.0 - abs(dot(-hitNormal, dir)), 5.0);

        if (pcg_hash() < d) {
            // specular ray
            float ddot = dot(dir, hitNormal);
            dir = dir - 2.0 * hitNormal * (ddot);

            float2 new_rand = float2(pcg_hash(), pcg_hash());

            float3 ggx_half_vector = ggx(new_rand, roughness);

            float3 new_norm = ggx_half_vector.x * tangent + ggx_half_vector.y * bitangent + ggx_half_vector.z * hitNormal;

            // TODO fix this monstrosity
            dir = reflect(dir, new_norm);
            dir = reflect(dir, hitNormal);

            if (dot(dir, hitNormal) <= 0.0) {
                break;
            }
            // if tracing specular, reset color - not true for metals
            current_color = float3(1.0);
        } else {
            float3 hemi = rand_hemi(float2(pcg_hash(), pcg_hash()));
            dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * hitNormal;
        }

        trace(offset_ray(pos, hitNormal), dir, payload);
        pos = payload.hit;
        hitNormal = payload.normal;

        // sky
        if (payload.isMiss) {
            float3 irradiance = intensity_indirect * bounce_energy * payload.color * env.sky_color_intensity.w * current_color;

            if (pc.indirect_intensity_clamp != 0.0) {
                irradiance = min(float3(pc.indirect_intensity_clamp), irradiance);
            }

            indirect += irradiance;
            break;
        } else {
            float3 irradiance = intensity_indirect * bounce_energy * lightSample(payload.hit, payload.normal, ray_shadow_dir, payload.color, payload);

            if (pc.indirect_intensity_clamp != 0.0) {
                irradiance = min(float3(pc.indirect_intensity_clamp), irradiance);
            }

            indirect += irradiance;
            intensity_indirect *= bounce_energy * current_color * payload.color;
        }
    }

    payload.isMiss = false;

    direct += lightSample(primaryPos, primaryNormal, ray_shadow_dir, primaryColor, payload);

    uint end = getRealtimeClockLow().x;

    if (global.debug == DebugMode::Direct) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                (direct),
                0.0
            )
        );
    } else if (global.debug == DebugMode::Indirect) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                (indirect),
                0.0
            )
        );
    } else if (global.debug == DebugMode::Time) {
        float time = float(time_diff(start, end)) / 1024 / 128;
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                time * time,
                -(pow(2.0 * time - 1.0, 2.0)) + 1.0,
                pow(1.0 - time, 2.0),
                0.0
            )
        );
    } else {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                (direct + indirect),
                0.0
            )
        );
    }
}


[shader("miss")]
void miss(inout Payload payload) {
    switch (skyVariant) {
    case SkyMode::Shader:
        payload.color = Toyrender::Sky::sample_distorted(WorldRayDirection(), images[pc.sky_idx]);
        break;
    case SkyMode::SingleColor:
        payload.color = env.sky_color_intensity.xyz;
        break;
    case SkyMode::Textured:
        payload.color = Toyrender::Sky::sample(WorldRayDirection(), images[pc.sky_idx], env.rotation);
        break;
    }

    payload.isMiss = true;
}


[shader("miss")]
void missEmpty(inout Payload payload) {
    payload.isMiss = true;
}


[shader("closesthit")]
void chit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.isMiss = false;
    payload.dist = RayTCurrent();

    float3x4 otw = ObjectToWorld();

    float3 vertPos0 = mul(otw, float4(HitTriangleVertexPosition(0), 1.0));
    float3 vertPos1 = mul(otw, float4(HitTriangleVertexPosition(1), 1.0));
    float3 vertPos2 = mul(otw, float4(HitTriangleVertexPosition(2), 1.0));

    float3 edgeA = vertPos1 - vertPos0;
    float3 edgeB = vertPos2 - vertPos0;

    //payload.normal = normalize(normal);
    payload.hit = attr.barycentrics.x * edgeA + attr.barycentrics.y * edgeB + vertPos0;

    let vertex0 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3];
    let vertex1 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3 + 1];
    let vertex2 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3 + 2];

    let color0 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex0][6]);
    let color1 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex1][6]);
    let color2 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex2][6]);

    let colorA = color1 - color0;
    let colorB = color2 - color0;

    payload.color = attr.barycentrics.x * colorA + attr.barycentrics.y * colorB + color0;

    let normal0 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex0][3]);
    let normal1 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex1][3]);
    let normal2 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex2][3]);

    float3 normalA = normal1 - normal0;
    float3 normalB = normal2 - normal0;

    let normalBlend = attr.barycentrics.x * normalA + attr.barycentrics.y * normalB + normal0;
    payload.normal = normalize(mul(float3x3(transpose(meshData[InstanceIndex()].model_inverse)), normalBlend));

    if (HitKind() == HIT_KIND_TRIANGLE_BACK_FACE) {
        payload.normal = -payload.normal;
    }

    //hash_state = InstanceID() * 4096;
    //payload.color = pow(float3(pcg_hash(), pcg_hash(), pcg_hash()), 2.2);

    //payload.color = float3(0.7);
}


[shader("closesthit")]
void chitEmpty(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.isMiss = false;
}

