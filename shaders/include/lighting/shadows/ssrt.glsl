#if !defined INCLUDE_LIGHTING_SHADOWS_SSRT_SHADOWS
#define INCLUDE_LIGHTING_SHADOWS_SSRT_SHADOWS

#include "/include/lighting/shadows/common.glsl"
#include "/include/utility/random.glsl"
#include "/include/utility/sampling.glsl"
#include "/include/utility/space_conversion.glsl"

bool raymarch_shadow(
    sampler2D depth_sampler,
    mat4 projection_matrix,
    mat4 projection_matrix_inverse,
    bool is_lod_depth,
    vec3 ray_origin_screen,
    vec3 ray_origin_view,
    vec3 ray_dir_view,
    bool has_sss,
    float dither,
    out float sss_depth
) {
    const uint step_count = uint(SHADOW_SSRT_STEPS);
    const float step_ratio = 2.0; // geometric sample distribution

    vec3 ray_dir_screen = normalize(
        view_to_screen_space(
            projection_matrix,
            ray_origin_view + ray_dir_view,
            true
        ) -
        ray_origin_screen
    );

    float ray_length = min_of(
        abs(sign(ray_dir_screen) - ray_origin_screen) /
        max(abs(ray_dir_screen), eps)
    );
    ray_length =
        min(ray_length,
            max(0.1, exp(-max0(length(ray_origin_view) * 0.025 - 1.0))));

    const float initial_step_scale = step_ratio == 1.0
        ? rcp(float(step_count))
        : (step_ratio - 1.0) / (pow(step_ratio, float(step_count)) - 1.0);
    float step_length = ray_length * initial_step_scale;

    vec3 ray_pos = ray_origin_screen + length(view_pixel_size) * ray_dir_screen;

    bool hit = false;
    bool hit_after_sss = false;
    bool sss_raymarch = has_sss;
    vec3 exit_pos = ray_origin_view;

    for (int i = 0; i < step_count; ++i) {
        step_length *= step_ratio;
        vec3 ray_step = ray_dir_screen * step_length;
        vec3 dithered_pos = ray_pos + dither * ray_step;
        ray_pos += ray_step;

#ifdef LOD_MOD_ACTIVE
        if (dithered_pos.z < 0.0) {
            continue;
        }
#endif
        if (clamp01(dithered_pos) != dithered_pos) {
            break;
        }

        ivec2 depth_size = textureSize(depth_sampler, 0);
        ivec2 depth_texel = clamp(
            ivec2(dithered_pos.xy * vec2(depth_size)),
            ivec2(0),
            depth_size - 1
        );
        float depth = texelFetch(depth_sampler, depth_texel, 0).x;

#ifdef LOD_MOD_ACTIVE
        if (is_lod_depth) {
            // Conservative 2x2 depth fetch reduces jagged distant LoD edges.
            ivec2 max_texel = textureSize(depth_sampler, 0) - 1;
            ivec2 t00 = clamp(depth_texel, ivec2(0), max_texel);
            ivec2 t10 = clamp(depth_texel + ivec2(1, 0), ivec2(0), max_texel);
            ivec2 t01 = clamp(depth_texel + ivec2(0, 1), ivec2(0), max_texel);
            ivec2 t11 = clamp(depth_texel + ivec2(1, 1), ivec2(0), max_texel);

            float d00 = texelFetch(depth_sampler, t00, 0).x;
            float d10 = texelFetch(depth_sampler, t10, 0).x;
            float d01 = texelFetch(depth_sampler, t01, 0).x;
            float d11 = texelFetch(depth_sampler, t11, 0).x;

            depth = min(min(d00, d10), min(d01, d11));
        }
#endif

        float z_ray = screen_to_view_space_depth(
            projection_matrix_inverse,
            dithered_pos.z
        );
        float z_sample =
            screen_to_view_space_depth(projection_matrix_inverse, depth);
        float z_delta = z_ray - z_sample;

        // LoD terrain depth is coarse and tends to self-intersect in SSRT.
        // Require a larger minimum thickness there to suppress shadow acne.
        float z_min_thickness = is_lod_depth ? 0.8 : 0.05;
        float z_max_thickness = is_lod_depth ? 24.0 : 10.0;

        bool inside = depth != 0.0 && depth < dithered_pos.z &&
            z_delta > z_min_thickness && z_delta < z_max_thickness;
        hit = inside || hit;

        if (sss_raymarch) {
            if (!inside) {
                exit_pos = dithered_pos;
                sss_raymarch = false;
            }
        }

        else if (!sss_raymarch) {
            hit_after_sss = inside || hit_after_sss;
            if (hit) {
                break;
            }
        }
    }

    exit_pos = screen_to_view_space(projection_matrix_inverse, exit_pos, true);
    sss_depth =
        hit_after_sss ? -1.0 : max0(distance(ray_origin_view, exit_pos) * 0.2);

    return hit;
}

float get_screen_space_shadows(
    vec2 position_screen_xy,
    vec3 position_view,
    float depth,
#ifdef LOD_MOD_ACTIVE
    bool is_lod_fragment,
    float depth_lod,
#endif
    float skylight,
    bool has_sss,
    inout float sss_depth
) {
    // Dithering for ray offset
    float dither = texelFetch(noisetex, ivec2(gl_FragCoord.xy) & 511, 0).b;

    // Slightly randomise ray direction to create soft shadows
    vec2 hash = hash2(gl_FragCoord.xy);
    vec3 ray_dir =
        normalize(view_light_dir + 0.03 * uniform_sphere_sample(hash));

#ifdef LOD_MOD_ACTIVE
    const float ssrt_reliable_distance = 640.0;
    const float ssrt_disable_distance = 1024.0;
    bool use_lod_ssrt = is_lod_fragment ||
        length_squared(position_view) > sqr(ssrt_reliable_distance);

    if (use_lod_ssrt) {
        // Disable temporal jitter on coarse LoD depth to reduce shimmer.
        dither = 0.0;
    } else {
        dither = r1(frameCounter, dither);
    }

    // Which depth map to raymarch depends on distance
    // Closer fragments: use combined depth texture (so MC terrain can cast)
    // Further fragments: use LoD depth texture (maximise precision)

    bool raymarch_combined_depth =
        !use_lod_ssrt &&
        length_squared(position_view) <
            sqr(far + 64.0); // heuristic of 4 chunks overlap
    bool hit;

    /*
    #ifdef MC_GL_KHR_shader_subgroup
    // Using subgroup ops, we make sure that if any fragments in a warp are
    raymarching the
    // combined depth buffer, they all do, to avoid divergent branches.
    raymarch_combined_depth = subgroupAny(raymarch_combined_depth);
    #endif
    */

    if (raymarch_combined_depth) {
        hit = raymarch_shadow(
            combined_depth_tex,
            combined_projection_matrix,
            combined_projection_matrix_inverse,
            false,
            vec3(position_screen_xy, depth),
            position_view,
            ray_dir,
            has_sss,
            dither,
            sss_depth
        );
    } else {
        vec3 lod_ray_origin_screen =
            view_to_screen_space(lod_projection_matrix, position_view, true);
        lod_ray_origin_screen.z = depth_lod;

        hit = raymarch_shadow(
            lod_depth_tex_solid,
            lod_projection_matrix,
            lod_projection_matrix_inverse,
            true,
            lod_ray_origin_screen,
            position_view,
            view_light_dir,
            has_sss,
            dither,
            sss_depth
        );
    }
#else
    dither = r1(frameCounter, dither);

    bool hit = raymarch_shadow(
        depthtex1,
        gbufferProjection,
        gbufferProjectionInverse,
        false,
        vec3(position_screen_xy, depth),
        position_view,
        view_light_dir,
        has_sss,
        dither,
        sss_depth
    );
#endif

    float ssrt_shadow = float(!hit) * get_lightmap_light_leak_prevention(skylight);

#ifdef LOD_MOD_ACTIVE
    if (use_lod_ssrt) {
        // Keep distant shadows while damping LoD SSRT aliasing.
        float lightmap_shadow = get_lightmap_shadows(skylight);
        float dist = length(position_view);
        float far_fade =
            linear_step(ssrt_reliable_distance, ssrt_disable_distance, dist);
        float ssrt_weight = mix(0.10, 0.0, far_fade);
        return mix(lightmap_shadow, ssrt_shadow, ssrt_weight);
    }
#endif

    return ssrt_shadow;
}

#endif // INCLUDE_LIGHTING_SHADOWS_SSRT_SHADOWS
