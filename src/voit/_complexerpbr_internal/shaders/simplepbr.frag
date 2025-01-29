// Based on code from https://github.com/KhronosGroup/glTF-Sample-Viewer

#version 300 es
precision highp float;

#ifndef MAX_LIGHTS
    #define MAX_LIGHTS 8
#endif

uniform struct p3d_MaterialParameters {
    vec4 baseColor;
    vec4 emission;
    float roughness;
    float metallic;
} p3d_Material;

uniform struct p3d_LightSourceParameters {
    vec4 position;
    vec4 diffuse;
    vec4 specular;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
#ifdef ENABLE_SHADOWS
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
#endif
} p3d_LightSource[MAX_LIGHTS];

uniform struct p3d_LightModelParameters {
    vec4 ambient;
} p3d_LightModel;

#ifdef ENABLE_FOG
uniform struct p3d_FogParameters {
    vec4 color;
    float density;
} p3d_Fog;
#endif

uniform vec4 p3d_ColorScale;
uniform vec4 p3d_TexAlphaOnly;

uniform vec3 sh_coeffs[9];
uniform vec3 camera_world_position;

struct FunctionParamters {
    float n_dot_l;
    float n_dot_v;
    float n_dot_h;
    float l_dot_h;
    float v_dot_h;
    float roughness;
    float metallic;
    vec3 reflection0;
    vec3 diffuse_color;
    vec3 specular_color;
};

uniform sampler2D p3d_TextureBaseColor;
uniform sampler2D p3d_TextureMetalRoughness;
uniform sampler2D p3d_TextureNormal;
uniform sampler2D p3d_TextureEmission;

uniform sampler2D brdf_lut;
uniform samplerCube filtered_env_map;
uniform float max_reflection_lod;

uniform mat4 CPUView2WorldCSMatrix;
uniform mat4 CPUWorld2Envmap;
uniform samplerCube skybox_tex;
uniform sampler2D reflection_color_and_alpha_tex;
uniform mat4 p3d_ProjectionMatrix;
uniform bool is_reflective_plane;
uniform mat4 reflective_plane_world_model_mat;
uniform bool from_reflective_cam;
uniform vec3 reflected_plane_local_normal;
uniform bool show_reflective_plane;
uniform bool reflective_plane_screen_texcoord;

#ifdef ENABLE_SHADOWS
uniform float global_shadow_bias;
#endif


const vec3 F0 = vec3(0.04);
const float PI = 3.141592653589793;
const float SPOTSMOOTH = 0.001;
const float LIGHT_CUTOFF = 0.001;

in vec3 v_view_position;
in vec3 v_world_position;
in vec4 v_color;
in vec2 v_texcoord;
in mat3 v_view_tbn;
in mat3 v_world_tbn;
#ifdef ENABLE_SHADOWS
in vec4 v_shadow_pos[MAX_LIGHTS];
#endif
in vec3 v_refl_view_pos;
in vec3 v_plane_normal;

out vec4 o_color;
layout(location=1) out vec4 out_reflection_ray_color_and_alpha;
layout(location=2) out vec4 out_reflection_weight_roughness;

////////////////////////// NON-REFLECTION PBR //////////////////////////////////

// Schlick's Fresnel approximation with Spherical Gaussian approximation to replace the power
vec3 specular_reflection(FunctionParamters func_params) {
    vec3 f0 = func_params.reflection0;
    float v_dot_h= func_params.v_dot_h;
    return f0 + (vec3(1.0) - f0) * pow(2.0, (-5.55473 * v_dot_h - 6.98316) * v_dot_h);
}

vec3 fresnelSchlickRoughness(float u, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - u, 0.0, 1.0), 5.0);
}

// Smith GGX with optional fast sqrt approximation (see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg))
float visibility_occlusion(FunctionParamters func_params) {
    float r = func_params.roughness;
    float n_dot_l = func_params.n_dot_l;
    float n_dot_v = func_params.n_dot_v;
#ifdef SMITH_SQRT_APPROX
    float ggxv = n_dot_l * (n_dot_v * (1.0 - r) + r);
    float ggxl = n_dot_v * (n_dot_l * (1.0 - r) + r);
#else
    float r2 = r * r;
    float ggxv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - r2) + r2);
    float ggxl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - r2) + r2);
#endif

    float ggx = ggxv + ggxl;
    if (ggx > 0.0) {
        return 0.5 / ggx;
    }
    return 0.0;
}

// GGX/Trowbridge-Reitz
float microfacet_distribution(FunctionParamters func_params) {
    float roughness2 = func_params.roughness * func_params.roughness;
    float f = (func_params.n_dot_h * func_params.n_dot_h) * (roughness2 - 1.0) + 1.0;
    return roughness2 / (PI * f * f);
}

// Lambert
float diffuse_function() {
    return 1.0 / PI;
}

#ifdef ENABLE_SHADOWS
float shadow_caster_contrib(sampler2DShadow shadowmap, vec4 shadowpos) {
    vec3 light_space_coords = shadowpos.xyz / shadowpos.w;
    light_space_coords.z -= global_shadow_bias;
    float shadow = texture(shadowmap, light_space_coords);
    return shadow;
}
#endif

vec3 get_normalmap_data(vec2 f_texcoord) {
#ifdef CALC_NORMAL_Z
    vec2 normalXY = 2.0 * texture(p3d_TextureNormal, f_texcoord).rg - 1.0;
    float normalZ = sqrt(clamp(1.0 - dot(normalXY, normalXY), 0.0, 1.0));
    return vec3(
        normalXY,
        normalZ
    );
#else
    return 2.0 * texture(p3d_TextureNormal, f_texcoord).rgb - 1.0;
#endif
}

vec3 irradiance_from_sh(vec3 normal) {
    return
        + sh_coeffs[0] * 0.282095
        + sh_coeffs[1] * 0.488603 * normal.x
        + sh_coeffs[2] * 0.488603 * normal.z
        + sh_coeffs[3] * 0.488603 * normal.y
        + sh_coeffs[4] * 1.092548 * normal.x * normal.z
        + sh_coeffs[5] * 1.092548 * normal.y * normal.z
        + sh_coeffs[6] * 1.092548 * normal.y * normal.x
        + sh_coeffs[7] * (0.946176 * normal.z * normal.z - 0.315392)
        + sh_coeffs[8] * 0.546274 * (normal.x * normal.x - normal.y * normal.y);
}

///////////////////////////// REFLECTION PBR /////////////////////////////



vec3 get_skybox(vec3 ray_dir_vs){
    vec3 ray_dir_cpuvs = vec3(ray_dir_vs.x, -ray_dir_vs.z, ray_dir_vs.y);
    vec3 ray_dir_cpuws = (CPUView2WorldCSMatrix*vec4(ray_dir_cpuvs, 1)).xyz;
    vec3 ray_dir_ws = vec3(ray_dir_cpuws.x, ray_dir_cpuws.z, -ray_dir_cpuws.y);
    vec3 ray_dir_cpuenvs = (CPUWorld2Envmap * vec4(ray_dir_cpuws, 1)).xyz;
    return texture(skybox_tex, ray_dir_cpuenvs).rgb;
}

vec4 get_reflection_color_and_alpha(vec3 pos_vs, vec3 normal_vs){
    vec4 q2 = vec4(v_refl_view_pos.x, v_refl_view_pos.z, -v_refl_view_pos.y, 1);
    vec4 refl_clip = p3d_ProjectionMatrix * q2;
    vec4 refl_ndc = refl_clip / refl_clip.w;
    vec2 refl_texcoord = refl_ndc.xy/2.0+0.5;
    refl_texcoord.y = 1.0- refl_texcoord.y;


    vec4 ray_test_result = texture(reflection_color_and_alpha_tex, refl_texcoord);

    if(ray_test_result.w < 1e-18){
        vec3 refl_dir_vs = reflect(normalize(pos_vs.xyz), normal_vs);
        ray_test_result = vec4(get_skybox(refl_dir_vs), 1);
    }

    return ray_test_result;
}

////////////////////////////// MAIN ///////////////////////////////

void main() {
    if( show_reflective_plane && (!is_reflective_plane)){
        vec3 pos_rp = (reflective_plane_world_model_mat*vec4(v_world_position, 1)).xyz;
        if(dot(normalize(reflected_plane_local_normal), normalize(pos_rp)) < 0.0){
            discard;
        }
    }

    vec2 f_texcoord = vec2(0);
    if(is_reflective_plane && reflective_plane_screen_texcoord){
        vec4 f_ndc = p3d_ProjectionMatrix*vec4(v_view_position, 1);
        f_ndc /= f_ndc.w;
        f_texcoord = f_ndc.xy/2.0+0.5;
    }
    else{
        f_texcoord = v_texcoord;
    }

    vec4 metal_rough = texture(p3d_TextureMetalRoughness, f_texcoord);
    float metallic = clamp(p3d_Material.metallic * metal_rough.b, 0.0, 1.0);
    float perceptual_roughness = clamp(p3d_Material.roughness * metal_rough.g,  0.0, 1.0);
    float alpha_roughness = perceptual_roughness * perceptual_roughness;
    vec4 base_color = p3d_Material.baseColor * v_color * p3d_ColorScale * (texture(p3d_TextureBaseColor, f_texcoord) + p3d_TexAlphaOnly);
    vec3 diffuse_color = (base_color.rgb * (vec3(1.0) - F0)) * (1.0 - metallic);
    vec3 spec_color = mix(F0, base_color.rgb, metallic);
#ifdef USE_NORMAL_MAP
    vec3 normalmap = get_normalmap_data();
#else
    vec3 normalmap = vec3(0, 0, 1);
#endif
    vec3 n = normalize(v_view_tbn * normalmap);
    vec3 world_normal = normalize(v_world_tbn * normalmap);
    vec3 v = normalize(-v_view_position);

#ifdef USE_OCCLUSION_MAP
    float ambient_occlusion = metal_rough.r;
#else
    float ambient_occlusion = 1.0;
#endif

#ifdef USE_EMISSION_MAP
    vec3 emission = p3d_Material.emission.rgb * texture(p3d_TextureEmission, f_texcoord).rgb;
#else
    vec3 emission = vec3(0.0);
#endif

    vec4 color = vec4(vec3(0.0), base_color.a);

    float n_dot_v = clamp(abs(dot(n, v)), 0.0, 1.0);

    for (int i = 0; i < p3d_LightSource.length(); ++i) {
        vec3 lightcol = p3d_LightSource[i].diffuse.rgb;

        if (dot(lightcol, lightcol) < LIGHT_CUTOFF) {
            continue;
        }

        vec3 light_pos = p3d_LightSource[i].position.xyz - v_view_position * p3d_LightSource[i].position.w;
        vec3 l = normalize(light_pos);
        vec3 h = normalize(l + v);
        float dist = length(light_pos);
        vec3 att_const = p3d_LightSource[i].attenuation;
        float attenuation_factor = 1.0 / (att_const.x + att_const.y * dist + att_const.z * dist * dist);
        float spotcos = dot(normalize(p3d_LightSource[i].spotDirection), -l);
        float spotcutoff = p3d_LightSource[i].spotCosCutoff;
        float shadowSpot = smoothstep(spotcutoff-SPOTSMOOTH, spotcutoff+SPOTSMOOTH, spotcos);
#ifdef ENABLE_SHADOWS
        float shadow_caster = shadow_caster_contrib(p3d_LightSource[i].shadowMap, v_shadow_pos[i]);
#else
        float shadow_caster = 1.0;
#endif
        float shadow = shadowSpot * shadow_caster * attenuation_factor;

        FunctionParamters func_params;
        func_params.n_dot_l = clamp(dot(n, l), 0.0, 1.0);
        func_params.n_dot_v = n_dot_v;
        func_params.n_dot_h = clamp(dot(n, h), 0.0, 1.0);
        func_params.l_dot_h = clamp(dot(l, h), 0.0, 1.0);
        func_params.v_dot_h = clamp(dot(v, h), 0.0, 1.0);
        func_params.roughness = alpha_roughness;
        func_params.metallic =  metallic;
        func_params.reflection0 = spec_color;
        func_params.diffuse_color = diffuse_color;
        func_params.specular_color = spec_color;

        vec3 F = specular_reflection(func_params);
        float V = visibility_occlusion(func_params); // V = G / (4 * n_dot_l * n_dot_v)
        float D = microfacet_distribution(func_params);

        vec3 diffuse_contrib = diffuse_color * diffuse_function();
        vec3 spec_contrib = vec3(F * V * D);
        color.rgb += func_params.n_dot_l * lightcol * (diffuse_contrib + spec_contrib) * shadow;
    }

    // Indirect diffuse + specular (IBL)
    vec3 ibl_f = fresnelSchlickRoughness(n_dot_v, spec_color, perceptual_roughness);
    vec3 ibl_kd = (1.0 - ibl_f) * (1.0 - metallic);
    vec3 ibl_diff = base_color.rgb * max(irradiance_from_sh((CPUWorld2Envmap*vec4(world_normal, 1)).xyz), 0.0) * diffuse_function();

    vec3 world_view = normalize(camera_world_position - v_world_position);
    vec3 ibl_r = reflect(-world_view, world_normal);
    vec2 env_brdf = texture(brdf_lut, vec2(n_dot_v, perceptual_roughness)).rg;
    if(is_reflective_plane){
        vec3 reflection_weight = (ibl_f * env_brdf.x + env_brdf.y) * ambient_occlusion;
        
        out_reflection_weight_roughness = vec4(reflection_weight, perceptual_roughness);
        out_reflection_ray_color_and_alpha = get_reflection_color_and_alpha(v_view_position, v_plane_normal);
        color.rgb += (ibl_kd * ibl_diff) * ambient_occlusion;
    }
    else{
        vec3 ibl_spec_color = textureLod(filtered_env_map, (CPUWorld2Envmap*vec4(ibl_r, 1)).xyz, perceptual_roughness * max_reflection_lod).rgb;
        vec3 ibl_spec = ibl_spec_color * (ibl_f * env_brdf.x + env_brdf.y);
        color.rgb += (ibl_kd * ibl_diff  + ibl_spec) * ambient_occlusion;
        out_reflection_ray_color_and_alpha = vec4(0, 0, 0, 0);
        out_reflection_weight_roughness = vec4(0, 0, 0, 0);
    }

    // Indirect diffuse (ambient light)
    color.rgb += (diffuse_color + spec_color) * p3d_LightModel.ambient.rgb * ambient_occlusion;

    // Emission
    color.rgb += emission;

#ifdef ENABLE_FOG
    // Exponential fog
    float fog_distance = length(v_view_position);
    float fog_factor = clamp(1.0 / exp(fog_distance * p3d_Fog.density), 0.0, 1.0);
    color = mix(p3d_Fog.color, color, fog_factor);
#endif

    o_color = color;
}
