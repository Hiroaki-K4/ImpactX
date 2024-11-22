#version 400 core
out vec4 FragColor;

in vec3 TexCoords;
uniform samplerCube spacebox;

void main() {
    FragColor = texture(spacebox, TexCoords);
}
