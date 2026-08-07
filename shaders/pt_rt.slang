import Common;
import Sampling;
import Sky;

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_shader_realtime_clock : enable
#extension GL_EXT_nonuniform_qualifier : enable

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
    int bounces;
    int color_idx;
    int depth_idx;
    int normal_idx;
    int direct_idx;
    int indirect_idx;
    int sky_idx;
    float direct_trace_distance;
    float indirect_trace_distance;
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

    TraceRay(
        tlas,
        RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xFF,
        0,
        0,
        0,
        ray,
        payload
    );
}

uint time_diff(uint startTime, uint endTime) {
    return endTime >= startTime ? (endTime-startTime) : (~0u-(startTime-endTime));
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

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

struct LightWeights {
    float3 intensity;
    float weight;
}

LightWeights sunWeight(float sunDot) {
    float3 intensity = env.sun_color_intensity.xyz * env.sun_color_intensity.w;
    float lumaIntensity = luminance(intensity);
    float weight = lumaIntensity;

    if (sunDot > 0.0 && lumaIntensity > 0.0) {
        weight = max(weight, 0.01);
    }

    LightWeights lw;
    lw.intensity = intensity;
    lw.weight = weight;

    return lw;
}

float3 lightSample(float3 pos, float3 normal, float3 dir_ray, float3 color, in Payload payload, out float3 light_dir) {
    float3 radiance = 0.0;

    float sunDot = max(dot(dir_ray, normal), 0.0);

    // select light
    let sunLw = sunWeight(sunDot);
    float sum = sunLw.weight;

    float2 rand_vars_sphere = float2(pcg_hash(), pcg_hash());
    float3 hemi = rand_hemi(rand_vars_sphere);

    for (int i = 0; i < env.point_lights; i++) {
        let light = env.lights[i];

        let light_main_dir = normalize(light.pos_radius.xyz - pos);

        float3 tangent, bitangent;
        compute_default_basis(light_main_dir, tangent, bitangent);
        let hemi_dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * light_main_dir;

        let light_pos = light.pos_radius.xyz + hemi_dir * light.pos_radius.w;
        let dir = normalize(light_pos - pos);
        let len = distance(light_pos, pos);
        let intensity = ((light.color_intensity.w * luminance(light.color_intensity.xyz)) / (len * len));
        let angle = max(dot(normal, dir), 0.0);
        sum += intensity * angle;
    }

    float rand_light = pcg_hash() * sum;

    if (rand_light > (sunLw.weight)) {
        sum = sunLw.weight;
        var light : Optional<PointLight> = none;
        float3 light_pos;
        float3 dir;
        float len;
        float angle;
        float intensity;

        for (int i = 0; i < env.point_lights; i++) {
            light = env.lights[i];

            let light_main_dir = normalize(light.value.pos_radius.xyz - pos);

            float3 tangent, bitangent;
            compute_default_basis(light_main_dir, tangent, bitangent);
            let hemi_dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * light_main_dir;

            light_pos = light.value.pos_radius.xyz + hemi_dir * light.value.pos_radius.w;

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

        float compensate = max(sum / max(intensity * angle, 0.000001), 0.001);

        float shadow = float(payload.isMiss);
        radiance = shadow * intensity * light.value.color_intensity.xyz * compensate;
        light_dir = ray.Direction;
    } else if (sunLw.weight > 0.0) {
        float compensate = max(sum / max(sunLw.weight, 0.000001), 0.001);
        if (sunDot > 0.0 && sunLw.weight > 0.0) {
            payload.isMiss = true;
            traceShadow(pos, dir_ray, payload);
        }

        float shadow = float(payload.isMiss);
        radiance = shadow * sunLw.intensity * compensate;
        light_dir = dir_ray;
    }

    return radiance;
}

float3 fresnel(float angle, float3 value) {
    return (pow(1.0 - angle, 5.0) * (1.0 - value) + value);
}

float GGX(float a2, float angle) {
    float p = ((a2 - 1.0) * angle * angle + 1.0);

    return a2 / (PI * p * p);
}

float GGX_Vis(float viewAngle, float lightAngle, float a2) {
    float a = viewAngle * sqrt(a2 + lightAngle * (lightAngle - a2 * lightAngle));
    float b = lightAngle * sqrt(a2 + viewAngle * (viewAngle - a2 * viewAngle));
    return 0.5f / (a + b);
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

    float2 sampleUv = (float2(launch_idx.xy) + 0.5) / float2(global.draw_res_x, global.draw_res_y);
    float2 uv = (float2(launch_idx.xy) + 0.5) / float2(global.res_x, global.res_y);

    float depth = storages[pc.depth_idx].Load(launch_idx.xy).x;

    if (depth == 0.0) {
        storages[pc.direct_idx].Store(
            int2(launch_idx.xy),
            float4(0.0)
        );
        storages[pc.indirect_idx].Store(
            int2(launch_idx.xy),
            float4(0.0)
        );
        return;
    }

    uint start = getRealtimeClockLow().x;

    float3 color = storages[pc.color_idx].Load(launch_idx.xy).xyz;
    float3 normal = normalize((storages[pc.normal_idx].Load(launch_idx.xy).xyz - 0.5) * 2.0);

    float4 per_pos = mul(vp.proj_inverse, float4(sampleUv * 2.0 - 1.0, depth, 1.0));

    per_pos /= per_pos.w;
    float3 vertPos = mul(vp.view_inverse, per_pos).xyz;

    float3 loc = transpose(vp.view_inverse)[3].xyz;
    float3 view_dir = normalize(vertPos - loc);
    float dis = distance(loc, vertPos);

    float3 indirect = float3(0.0);

    float3 tangent, bitangent;

    // setup hasher with min index 1 so that even first sample is noisy
    hash_state = (launch_idx.x + launch_idx.y * uint(global.res_x)) * (global.frame_index + 1);

    float2 rand = float2(pcg_hash(), pcg_hash());

    compute_default_basis(env.sun_dir_sun_angle.xyz, tangent, bitangent);

    float3 cone = rand_cone(rand, cos(env.sun_dir_sun_angle.w));
    float3 ray_shadow_dir = cone.x * tangent + cone.y * bitangent + cone.z * env.sun_dir_sun_angle.xyz;

    float3 intensity_indirect = 1.0;
    float3 pos = vertPos;
    float3 hitNormal = normal;
    float3 dir = view_dir;

    // current surface_color
    float3 current_color = color;

    float3 direct = 0.0;

    float metallic = 0.0;

    float roughness = 1.0;
    for (int i = 0; i <= pc.bounces; i++) {
        compute_default_basis(hitNormal, tangent, bitangent);

        float d = roughness + (1.0 - roughness) * pow(1.0 - abs(dot(-hitNormal, dir)), 5.0);
        d = lerp(d, 1.0, metallic);

        bool specular = pcg_hash() < d;

        float3 dirInc = dir;
        float3 specColor = lerp(float3(1.0), current_color, metallic);
        // TODO replace for proper BRDF result
        float3 bounceColor = lerp(float3(1.0), current_color, specular);

        float2 new_rand = float2(pcg_hash(), pcg_hash());
        if (specular) {
            float3 ggx_half_vector = ggx(new_rand, roughness * roughness);

            float3 new_norm = ggx_half_vector.x * tangent + ggx_half_vector.y * bitangent + ggx_half_vector.z * hitNormal;
            dir = reflect(dir, new_norm);

            if (dot(dir, hitNormal) <= 0.0) {
                break;
            }
        } else {
            float3 hemi = rand_hemi(new_rand);
            dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * hitNormal;
        }

        float3 lightDir = 0.0;

        float3 radiance = intensity_indirect * lightSample(pos, hitNormal, ray_shadow_dir, float3(1.0), payload, lightDir);
        float lightNormalDot = dot(hitNormal, lightDir);

        lightNormalDot = max(lightNormalDot, 0.000001);

        float3 newHalf = normalize((lightDir - dirInc));
        float lightHalfDot = max(dot(lightDir, newHalf), 0.000001);
        float normalHalfDot = max(dot(hitNormal, newHalf), 0.000001);
        float normalViewDot = max(dot(hitNormal, -dirInc), 0.000001);
        float alpha = roughness * roughness;
        float a2 = alpha * alpha;

        float3 f0 = lerp(0.04, current_color, metallic);
        float3 F = fresnel(lightHalfDot, f0);

        float3 diffuseLobe = lightNormalDot * (current_color / PI) * (1.0 - metallic);
        float3 specularLobe = F * GGX(a2, normalHalfDot) * GGX_Vis(normalViewDot, lightNormalDot, a2) * lightNormalDot;

        radiance *= diffuseLobe + specularLobe;

        if (pc.indirect_intensity_clamp > 0.0) {
            float luma = max(luminance(radiance), 0.0001);
            radiance *= min(pc.indirect_intensity_clamp / luma, 1.0);
        }

        direct += radiance;
        intensity_indirect *= current_color;

        trace(offset_ray(pos, hitNormal), dir, payload);

        if (payload.isMiss) {
            break;
        } else {
            pos = payload.hit;
            hitNormal = payload.normal;
        }
    }

    compute_default_basis(hitNormal, tangent, bitangent);
    float3 hemi = rand_hemi(float2(pcg_hash(), pcg_hash()));
    dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * hitNormal;

    trace(offset_ray(pos, hitNormal), dir, payload);

    if (payload.isMiss) {
        indirect = 1.0;
    }

    uint end = getRealtimeClockLow().x;

    if (global.debug == DebugMode::Time) {
        float time = float(time_diff(start, end)) / 1024 / 128;
        storages[pc.direct_idx].Store(
            int2(launch_idx.xy),
            float4(
                time * time,
                -(pow(2.0 * time - 1.0, 2.0)) + 1.0,
                pow(1.0 - time, 2.0),
                0.0
            )
        );
    } else {
        storages[pc.direct_idx].Store(
            int2(launch_idx.xy),
            float4(
                direct,
                1.0
            )
        );
        storages[pc.indirect_idx].Store(
            int2(launch_idx.xy),
            float4(
                indirect,
                1.0
            )
        );
    }
}


[shader("miss")]
void miss(inout Payload payload) {
    payload.isMiss = true;

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

    if (pc.indirect_intensity_clamp != 0.0) {
        payload.color = min(float3(pc.indirect_intensity_clamp), payload.color);
    }
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

    // payload.normal = normalize(normal);
    payload.hit = attr.barycentrics.x * edgeA + attr.barycentrics.y * edgeB + vertPos0;

    float3 normal = cross(vertPos1 - vertPos0, vertPos2 - vertPos0);

    let vertex0 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3];
    let vertex1 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3 + 1];
    let vertex2 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3 + 2];

    payload.color = meshData[InstanceIndex()].color_rough.xyz;

    payload.normal = normalize(normal);
}
