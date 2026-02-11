#version 450
#pragma shader_stage(compute)
#extension GL_EXT_nonuniform_qualifier : enable

#include "common/debug_modes.glsl"
#include "common/defs.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

#define PI 3.14159265359

layout(set = 0, binding = 0) GLOBAL;
layout(set = 0, binding = 2) VIEW_PROJ;
layout(set = 0, binding = 3) ENV;

layout(set = 1, binding = 0) uniform sampler2D images[];
layout(set = 1, binding = 1, rgba16f) uniform image2D storages[];

layout(push_constant) uniform constants {
    int out_idx;
} push_consts;

#define PLANET_POS vec3(0.0) /* the position of the planet */
#define PLANET_RADIUS 6371e3 /* radius of the planet */
#define ATMOS_RADIUS 6471e3 /* radius of the atmosphere */
// scattering coeffs
#define RAY_BETA vec3(5.5e-6, 13.0e-6, 22.4e-6) /* rayleigh, affects the color of the sky */
#define MIE_BETA vec3(21e-6) /* mie, affects the color of the blob around the sun */
#define AMBIENT_BETA vec3(0.0) /* ambient, affects the scattering color when there is no lighting from the sun */
#define ABSORPTION_BETA vec3(2.04e-5, 4.97e-5, 1.95e-6) /* what color gets absorbed by the atmosphere (Due to things like ozone) */
#define G 0.7 /* mie scattering direction, or how big the blob around the sun is */
// and the heights (how far to go up before the scattering has no effect)
#define HEIGHT_RAY 8e3 /* rayleigh height */
#define HEIGHT_MIE 1.2e3 /* and mie */
#define HEIGHT_ABSORPTION 30e3 /* at what height the absorption is at it's maximum */
#define ABSORPTION_FALLOFF 4e3 /* how much the absorption decreases the further away it gets from the maximum height */
// and the steps (more looks better, but is slower)
// the primary step has the most effect on looks
#define PRIMARY_STEPS 32 /* primary steps, affects quality the most */
#define LIGHT_STEPS 8 /* light steps, how much steps in the light direction are taken */

/*
Next we'll define the main scattering function.
This traces a ray from start to end and takes a certain amount of samples along this ray, in order to calculate the color.
For every sample, we'll also trace a ray in the direction of the light,
because the color that reaches the sample also changes due to scattering
*/
vec3 calculate_scattering(
vec3 start, // the start of the ray (the camera position)
vec3 dir, // the direction of the ray (the camera vector)
vec3 light_dir, // the direction of the light
vec3 light_intensity, // how bright the light is, affects the brightness of the atmosphere
vec3 planet_position, // the position of the planet
float planet_radius, // the radius of the planet
float atmo_radius, // the radius of the atmosphere
vec3 beta_ray, // the amount rayleigh scattering scatters the colors (for earth: causes the blue atmosphere)
vec3 beta_mie, // the amount mie scattering scatters colors
vec3 beta_absorption, // how much air is absorbed
vec3 beta_ambient, // the amount of scattering that always occurs, cna help make the back side of the atmosphere a bit brighter
float g, // the direction mie scatters the light in (like a cone). closer to -1 means more towards a single direction
float height_ray, // how high do you have to go before there is no rayleigh scattering?
float height_mie, // the same, but for mie
float height_absorption, // the height at which the most absorption happens
float absorption_falloff, // how fast the absorption falls off from the absorption height
int steps_i, // the amount of steps along the 'primary' ray, more looks better but slower
int steps_l// the amount of steps along the light ray, more looks better but slower
) {
    // add an offset to the camera position, so that the atmosphere is in the correct position
    start -= planet_position;
    // calculate the start and end position of the ray, as a distance along the ray
    // we do this with a ray sphere intersect
    float a = dot(dir, dir);
    float b = 2.0 * dot(dir, start);
    float c = dot(start, start) - (atmo_radius * atmo_radius);
    float d = (b * b) - 4.0 * a * c;

    // stop early if there is no intersect
    if (d < 0.0) return vec3(0.0);

    // calculate the ray length
    vec2 ray_length = vec2(
    max((-b - sqrt(d)) / (2.0 * a), 0.0),
    min((-b + sqrt(d)) / (2.0 * a), 100000000.0)
    );

    // if the ray did not hit the atmosphere, return a black color
    if (ray_length.x > ray_length.y) return vec3(0.0);
    // prevent the mie glow from appearing if there's an object in front of the camera
    bool allow_mie = 100000000.0 > ray_length.y;
    // make sure the ray is no longer than allowed
    ray_length.y = min(ray_length.y, 100000000.0);
    ray_length.x = max(ray_length.x, 0.0);
    // get the step size of the ray
    float step_size_i = (ray_length.y - ray_length.x) / float(steps_i);

    // next, set how far we are along the ray, so we can calculate the position of the sample
    // if the camera is outside the atmosphere, the ray should start at the edge of the atmosphere
    // if it's inside, it should start at the position of the camera
    // the min statement makes sure of that
    float ray_pos_i = ray_length.x + step_size_i * 0.5;

    // these are the values we use to gather all the scattered light
    vec3 total_ray = vec3(0.0);// for rayleigh
    vec3 total_mie = vec3(0.0);// for mie

    // initialize the optical depth. This is used to calculate how much air was in the ray
    vec3 opt_i = vec3(0.0);

    // also init the scale height, avoids some vec2's later on
    vec2 scale_height = vec2(height_ray, height_mie);

    // Calculate the Rayleigh and Mie phases.
    // This is the color that will be scattered for this ray
    // mu, mumu and gg are used quite a lot in the calculation, so to speed it up, precalculate them
    float mu = dot(dir, light_dir);
    float mumu = mu * mu;
    float gg = g * g;
    float phase_ray = 3.0 / (50.2654824574 /* (16 * pi) */) * (1.0 + mumu);
    float phase_mie = allow_mie ? 3.0 / (25.1327412287 /* (8 * pi) */) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg)) : 0.0;

    // now we need to sample the 'primary' ray. this ray gathers the light that gets scattered onto it
    for (int i = 0; i < steps_i; ++i) {

        // calculate where we are along this ray
        vec3 pos_i = start + dir * ray_pos_i;

        // and how high we are above the surface
        float height_i = length(pos_i) - planet_radius;

        // now calculate the density of the particles (both for rayleigh and mie)
        vec3 density = vec3(exp(-height_i / scale_height), 0.0);

        // and the absorption density. this is for ozone, which scales together with the rayleigh,
        // but absorbs the most at a specific height, so use the sech function for a nice curve falloff for this height
        // clamp it to avoid it going out of bounds. This prevents weird black spheres on the night side
        float denom = (height_absorption - height_i) / absorption_falloff;
        density.z = (1.0 / (denom * denom + 1.0)) * density.x;

        // multiply it by the step size here
        // we are going to use the density later on as well
        density *= step_size_i;

        // Add these densities to the optical depth, so that we know how many particles are on this ray.
        opt_i += density;

        // Calculate the step size of the light ray.
        // again with a ray sphere intersect
        // a, b, c and d are already defined
        a = dot(light_dir, light_dir);
        b = 2.0 * dot(light_dir, pos_i);
        c = dot(pos_i, pos_i) - (atmo_radius * atmo_radius);
        d = (b * b) - 4.0 * a * c;

        // no early stopping, this one should always be inside the atmosphere
        // calculate the ray length
        float step_size_l = (-b + sqrt(d)) / (2.0 * a * float(steps_l));

        // and the position along this ray
        // this time we are sure the ray is in the atmosphere, so set it to 0
        float ray_pos_l = step_size_l * 0.5;

        // and the optical depth of this ray
        vec3 opt_l = vec3(0.0);

        // now sample the light ray
        // this is similar to what we did before
        for (int l = 0; l < steps_l; ++l) {

            // calculate where we are along this ray
            vec3 pos_l = pos_i + light_dir * ray_pos_l;

            // the heigth of the position
            float height_l = length(pos_l) - planet_radius;

            // calculate the particle density, and add it
            // this is a bit verbose
            // first, set the density for ray and mie
            vec3 density_l = vec3(exp(-height_l / scale_height), 0.0);

            // then, the absorption
            float denom = (height_absorption - height_l) / absorption_falloff;
            density_l.z = (1.0 / (denom * denom + 1.0)) * density_l.x;

            // multiply the density by the step size
            density_l *= step_size_l;

            // and add it to the total optical depth
            opt_l += density_l;

            // and increment where we are along the light ray.
            ray_pos_l += step_size_l;

        }

        // Now we need to calculate the attenuation
        // this is essentially how much light reaches the current sample point due to scattering
        vec3 attn = exp(-beta_ray * (opt_i.x + opt_l.x) - beta_mie * (opt_i.y + opt_l.y) - beta_absorption * (opt_i.z + opt_l.z));

        // accumulate the scattered light (how much will be scattered towards the camera)
        total_ray += density.x * attn;
        total_mie += density.y * attn;

        // and increment the position on this ray
        ray_pos_i += step_size_i;

    }

    // calculate and return the final color
    return (
    phase_ray * beta_ray * total_ray// rayleigh color
    + phase_mie * beta_mie * total_mie// and ambient
    ) * light_intensity;// now make sure the background is rendered correctly
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    ivec2 size = imageSize(storages[push_consts.out_idx]);

    if (x <= size.x && y <= size.y) {
        vec4 color = vec4(0.0);

        float vert = (float(x) / float(size.x)) * 2.0 * PI;
        float hor = (float(y) / float(size.y)) * 2.0 - 1.0;
        hor = sin(hor * (PI / 2.0));
        hor = hor * hor * sign(hor);
        float vert_intensity = sqrt(1.0 - (hor * hor));

        vec3 camera_vector = normalize(vec3(-cos(vert) * vert_intensity, -sin(vert) * vert_intensity, hor));

        vec3 camera_position = vec3(0.0, 0.0, PLANET_RADIUS + 100.0);

        // the color of this pixel
        vec3 col = calculate_scattering(
        camera_position, // the position of the camera
        camera_vector, // the camera vector (ray direction of this pixel)
        env.sun_dir_sun_angle.xyz, // light direction
        vec3(40.0), // light intensity, 40 looks nice
        PLANET_POS, // position of the planet
        PLANET_RADIUS, // radius of the planet in meters
        ATMOS_RADIUS, // radius of the atmosphere in meters
        RAY_BETA, // Rayleigh scattering coefficient
        MIE_BETA, // Mie scattering coefficient
        ABSORPTION_BETA, // Absorbtion coefficient
        AMBIENT_BETA, // ambient scattering, turned off for now. This causes the air to glow a bit when no light reaches it
        G, // Mie preferred scattering direction
        HEIGHT_RAY, // Rayleigh scale height
        HEIGHT_MIE, // Mie scale height
        HEIGHT_ABSORPTION, // the height at which the most absorption happens
        ABSORPTION_FALLOFF, // how fast the absorption falls off from the absorption height
        PRIMARY_STEPS, // steps in the ray direction
        LIGHT_STEPS// steps in the light direction
        );

        col = max(col, vec3(0.0));

        imageStore(storages[push_consts.out_idx], ivec2(x, y), vec4(col, 1.0));
    }
}
