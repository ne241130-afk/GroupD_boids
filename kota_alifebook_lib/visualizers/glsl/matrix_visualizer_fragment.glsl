uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main()
{
    vec3 color = texture2D(u_texture, v_texcoord).rgb;
    gl_FragColor = vec4(color, 1.0);
}
