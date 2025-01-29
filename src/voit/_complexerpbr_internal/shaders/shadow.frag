#version 300 es
precision highp float;

uniform struct p3d_MaterialParameters {
    vec4 baseColor;
} p3d_Material;

uniform vec4 p3d_ColorScale;

uniform sampler2D p3d_TextureBaseColor;
in vec4 v_color;
in vec2 v_texcoord;

out vec4 o_color;

void main() {
    vec4 base_color = p3d_Material.baseColor * v_color * p3d_ColorScale * texture(p3d_TextureBaseColor, v_texcoord);
    o_color = base_color;
}
