#version 300 es
precision highp float;

uniform mat4 p3d_ModelViewProjectionMatrix;
#ifdef ENABLE_SKINNING
uniform mat4 p3d_TransformTable[100];
#endif

in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
#ifdef ENABLE_SKINNING
in vec4 transform_weight;
in vec4 transform_index;
#endif


out vec4 v_color;
out vec2 v_texcoord;

void main() {
#ifdef ENABLE_SKINNING
    mat4 skin_matrix = (
        p3d_TransformTable[int(transform_index.x)] * transform_weight.x +
        p3d_TransformTable[int(transform_index.y)] * transform_weight.y +
        p3d_TransformTable[int(transform_index.z)] * transform_weight.z +
        p3d_TransformTable[int(transform_index.w)] * transform_weight.w
    );
    vec4 vert_pos4 = skin_matrix * p3d_Vertex;
#else
    vec4 vert_pos4 = p3d_Vertex;
#endif
    v_color = p3d_Color;
    v_texcoord = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * vert_pos4;
}
