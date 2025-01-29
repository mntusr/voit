#version 300 es
precision highp float;

uniform sampler2D tex;
#ifdef USE_SDR_LUT
    uniform sampler3D sdr_lut;
    uniform float sdr_lut_factor;
#endif
uniform float exposure;
uniform mat4 orig_CPUProjectionMatrixInverse;
uniform mat4 orig_CPUView2WorldCSMatrix;
uniform mat4 orig_CPUWorld2Envmap;
uniform vec2 texture_size;

uniform samplerCube skybox_tex;
uniform sampler2D reflected_color_tex;
uniform sampler2D reflection_weight_and_roughness_tex;
uniform bool show_envmap_as_skybox;

in vec2 v_texcoord;

out vec4 o_color;

///////////////////// UTILITY FUNCTIONS /////////////////

vec3 get_regular_color(vec2 pos_uv){
    return texture(tex, pos_uv).rgb;
}
vec3 get_reflection_color(vec2 pos_uv){
    return texture(reflected_color_tex, pos_uv).rgb;
}
vec3 get_reflection_weight(vec2 pos_uv){
    return texture(reflection_weight_and_roughness_tex, pos_uv).rgb;
}
float get_roughness(vec2 pos_uv){
    return texture(reflection_weight_and_roughness_tex, pos_uv).w;
}

vec3 get_skybox(vec3 ray_dir_vs){
    vec3 ray_dir_cpuvs = vec3(ray_dir_vs.x, -ray_dir_vs.z, ray_dir_vs.y);
    vec3 ray_dir_cpuws = (orig_CPUView2WorldCSMatrix*vec4(ray_dir_cpuvs, 1)).xyz;
    vec3 ray_dir_ws = vec3(ray_dir_cpuws.x, ray_dir_cpuws.z, -ray_dir_cpuws.y);
    vec3 ray_dir_cpuenvs = (orig_CPUWorld2Envmap * vec4(ray_dir_cpuws, 1)).xyz;
    return texture(skybox_tex, ray_dir_cpuenvs).rgb;
}


vec3 get_ray_dir_from_uv_on_near_plane(vec2 pos_uv){
    vec4 pos_ndc = vec4(pos_uv*2.0-1.0, -1, 1);
    vec4 pos_cpuvs = orig_CPUProjectionMatrixInverse*pos_ndc;
    pos_cpuvs /= pos_cpuvs.w;
    vec3 pos_vs = vec3(pos_cpuvs.x, pos_cpuvs.z, -pos_cpuvs.y);
    return normalize(pos_vs);
}
bool is_sky(ivec2 pos_fg){
    return texelFetch(tex, pos_fg, 0).w < 1e-18;
}
bool is_reflection(ivec2 pos_fg){
    return texelFetch(reflected_color_tex, pos_fg, 0).w > 1e-18;
}
vec2 fg2uv(vec2 fg){
    return fg / texture_size;
}
vec2 uv2fg(vec2 uv){
    return uv * texture_size;
}


/////////////////////// REFLECTION //////////////////////////

vec3 get_reflection_color_with_roughness(vec2 pos_uv, float roughness){
    float d_fg = roughness*10.0;
    vec2 pos_fg = uv2fg(pos_uv);

    int denom = 0;
    vec3 color_acc = vec3(0.0);
    
    for(float i_x=-1.0; i_x<2.0; i_x++){
        for(float i_y=-1.0; i_y<2.0; i_y++){
            vec2 sample_pos_fg = vec2(pos_fg.x + i_x*d_fg, pos_fg.y+i_y*d_fg);

            if(is_reflection(ivec2(sample_pos_fg))){
                vec2 sample_pos_uv = fg2uv(sample_pos_fg);
                if(
                        (sample_pos_uv.x < 1.0) && 
                        (sample_pos_uv.x > 0.0) && 
                        (sample_pos_uv.y < 1.0) && 
                        (sample_pos_uv.y > 0.0)){
                    denom += 1;
                    color_acc += get_reflection_color(sample_pos_uv);
                }
            }
        }
    }

    return color_acc / (float(denom)+1e-18);
}

////////////////////////// TONEMAPPING /////////////////////////////

vec3 apply_tonemapping(vec3 color){
    color *= exposure;
    color = max(vec3(0.0), color - vec3(0.004));
    color = (color * (vec3(6.2) * color + vec3(0.5))) / (color * (vec3(6.2) * color + vec3(1.7)) + vec3(0.06));

#ifdef USE_SDR_LUT
    vec3 lut_size = vec3(textureSize(sdr_lut, 0));
    vec3 lut_uvw = (color.rgb * float(lut_size - 1.0) + 0.5) / lut_size;
    vec3 lut_color = texture(sdr_lut, lut_uvw).rgb;
    color = mix(color, lut_color, sdr_lut_factor);
#endif
    return color;
}

///////////////////////////// MAIN PART //////////////////////////////

void main() {
    vec3 total_color = vec3(0);
    vec2 texel_fg = uv2fg(v_texcoord);
    if(show_envmap_as_skybox && is_sky(ivec2(texel_fg))){
        vec3 frag_dir_vs = get_ray_dir_from_uv_on_near_plane(v_texcoord);
        total_color = get_skybox(frag_dir_vs);
    }
    else{
        float roughness = get_roughness(v_texcoord);
        vec3 reflected_color = get_reflection_color_with_roughness(v_texcoord, roughness);
        vec3 regular_color = get_regular_color(v_texcoord);
        vec3 reflection_weight = get_reflection_weight(v_texcoord);

        total_color = regular_color + reflected_color*reflection_weight;
    }

    total_color = apply_tonemapping(total_color);

    o_color = vec4(total_color, 1.0);
}
