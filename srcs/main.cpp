#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.hpp"
#include "Particle.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


glm::vec3 cameraPos = glm::vec3(0.0f, 3.0f, 12.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

float yaw = -90.0f, pitch = 0.0f;
float lastX = 400, lastY = 300;
bool firstMouse = true;
float fov = 45.0f;

const int LAT_SEGMENTS = 100;
const int LON_SEGMENTS = 100;
const float PI = 3.14159265359f;

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    float cameraSpeed = 2.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPos += cameraSpeed * cameraFront;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cameraPos -= cameraSpeed * cameraFront;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        cameraPos += cameraSpeed * cameraUp;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        cameraPos -= cameraSpeed * cameraUp;
    }
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;
    if (pitch > 89.0f) {
        pitch = 89.0f;
    }
    if (pitch < -89.0f) {
        pitch = -89.0f;
    }

    glm::vec3 direction;
    direction.x = cos(glm::radians(yaw) * cos(glm::radians(pitch)));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(direction);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f) {
        fov = 1.0f;
    }
    if (fov > 45.0f) {
        fov = 45.0f;
    }
}

unsigned int loadTexture(char const * path, bool repeat) {
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1) {
            format = GL_RED;
        } else if (nrComponents == 3) {
            format = GL_RGB;
        } else if (nrComponents == 4) {
            format = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        if (repeat) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        } else {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    } else {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

unsigned int loadCubemap(std::vector<std::string> faces) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        } else {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

std::vector<float> generateParticleVertices(float radius) {
    std::vector<float> vertices;

    for (int lat = 0; lat <= LAT_SEGMENTS; ++lat) {
        int fix_lat = lat - int(LAT_SEGMENTS / 2);
        float phi = fix_lat * PI / LAT_SEGMENTS;
        float sinPhi = sin(phi);
        float cosPhi = cos(phi);

        for (int lon = 0; lon <= LON_SEGMENTS; ++lon) {
            float theta = lon * 2 * PI / LON_SEGMENTS;
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);

            float x = radius * cosPhi * cosTheta;
            float y = radius * sinPhi;
            float z = radius * cosPhi * sinTheta;

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    return vertices;
}

int main(int argc, char *argv[]) {
    // Initialize window
    if (!glfwInit()) {
        std::cout << "Error: Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int window_w = 1920;
    int window_h = 1080;
    std::string window_title = "NewtonX";
    GLFWwindow* window = glfwCreateWindow(window_w, window_h, window_title.c_str(), NULL, NULL);
    if (window == NULL) {
        std::cout << "Error: Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Error: Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    Shader shader("../shaders/floor.vs", "../shaders/floor.fs");
    Shader particle_shader("../shaders/particle.vs", "../shaders/particle.fs");
    Shader space_box_shader("../shaders/space_box.vs", "../shaders/space_box.fs");

    float cubeVertices[] = {
        // positions          // texture Coords
        // Front face
        -0.25f, -0.25f, -0.25f,  1.0f, 1.0f,
        0.25f, -0.25f, -0.25f,  0.0f, 1.0f,
        0.25f,  0.25f, -0.25f,  0.0f, 0.0f,
        0.25f,  0.25f, -0.25f,  0.0f, 0.0f,
        -0.25f,  0.25f, -0.25f,  1.0f, 0.0f,
        -0.25f, -0.25f, -0.25f,  1.0f, 1.0f,

        // Back face
        -0.25f, -0.25f,  0.25f,  1.0f, 1.0f,
        0.25f, -0.25f,  0.25f,  0.0f, 1.0f,
        0.25f,  0.25f,  0.25f,  0.0f, 0.0f,
        0.25f,  0.25f,  0.25f,  0.0f, 0.0f,
        -0.25f,  0.25f,  0.25f,  1.0f, 0.0f,
        -0.25f, -0.25f,  0.25f,  1.0f, 1.0f,

        // Left face
        -0.25f, -0.25f,  0.25f,  1.0f, 1.0f,
        -0.25f, -0.25f, -0.25f,  0.0f, 1.0f,
        -0.25f,  0.25f, -0.25f,  0.0f, 0.0f,
        -0.25f,  0.25f, -0.25f,  0.0f, 0.0f,
        -0.25f,  0.25f,  0.25f,  1.0f, 0.0f,
        -0.25f, -0.25f,  0.25f,  1.0f, 1.0f,

        // Right face
        0.25f, -0.25f,  0.25f,  1.0f, 1.0f,
        0.25f, -0.25f, -0.25f,  0.0f, 1.0f,
        0.25f,  0.25f, -0.25f,  0.0f, 0.0f,
        0.25f,  0.25f, -0.25f,  0.0f, 0.0f,
        0.25f,  0.25f,  0.25f,  1.0f, 0.0f,
        0.25f, -0.25f,  0.25f,  1.0f, 1.0f,

        // Bottom face
        -0.25f, -0.25f, -0.25f,  1.0f, 1.0f,
        0.25f, -0.25f, -0.25f,  0.0f, 1.0f,
        0.25f, -0.25f,  0.25f,  0.0f, 0.0f,
        0.25f, -0.25f,  0.25f,  0.0f, 0.0f,
        -0.25f, -0.25f,  0.25f,  1.0f, 0.0f,
        -0.25f, -0.25f, -0.25f,  1.0f, 1.0f,

        // Top face
        -0.25f,  0.25f, -0.25f,  1.0f, 1.0f,
        0.25f,  0.25f, -0.25f,  0.0f, 1.0f,
        0.25f,  0.25f,  0.25f,  0.0f, 0.0f,
        0.25f,  0.25f,  0.25f,  0.0f, 0.0f,
        -0.25f,  0.25f,  0.25f,  1.0f, 0.0f,
        -0.25f,  0.25f, -0.25f,  1.0f, 1.0f
    };

    float space_box_vertices[] = {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
        1.0f,  1.0f, -1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
        1.0f, -1.0f,  1.0f
    };

    std::vector<float> particle_vertices = generateParticleVertices(0.3f);
    int particle_num = 4;
    Particle particles = Particle(particle_num);

    // Initialize window
    glViewport(0, 0, window_w, window_h);

    // Store instance data in an array buffer
    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * particle_num, particles.get_particle_position().data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Cube VAO
    unsigned int cubeVAO, cubeVBO;
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), &cubeVertices, GL_STATIC_DRAW);
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // particle VAO
    unsigned int particleVAO, particleVBO;
    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);
    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, particle_vertices.size() * sizeof(float), particle_vertices.data(), GL_STATIC_DRAW);
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // Space box VAO
    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(space_box_vertices), &space_box_vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    // set instance data(position)
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribDivisor(1, 1); // tell OpenGL this is an instanced vertex attribute

    glEnable(GL_DEPTH_TEST);

    unsigned int cubeTexture = loadTexture("../textures/awesomeface.png", true);

    std::vector<std::string> faces
    {
        "../textures/space_box/right.png",
        "../textures/space_box/left.png",
        "../textures/space_box/top.png",
        "../textures/space_box/bottom.png",
        "../textures/space_box/front.png",
        "../textures/space_box/back.png"
    };
    unsigned int cubemapTexture = loadCubemap(faces);

    double last_time = glfwGetTime();
    double fps_last_time = glfwGetTime();
    int frame_num = 0;
    shader.use();
    shader.setInt("texture1", 0);
    space_box_shader.use();
    space_box_shader.setInt("spacebox", 0);
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        // glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov), float(window_w) / float(window_h), 0.1f, 100.0f);
        unsigned int viewLoc = glGetUniformLocation(shader.ID, "view");
        unsigned int projLoc = glGetUniformLocation(shader.ID, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);

        // Cube
        glBindVertexArray(cubeVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cubeTexture);
        model = glm::translate(model, glm::vec3(-1.0f, 0.0f, -1.0f));
        unsigned int modeLoc = glGetUniformLocation(shader.ID, "model");
        glUniformMatrix4fv(modeLoc, 1, GL_FALSE, &model[0][0]);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(2.0f, 0.0f, 0.0f));
        glUniformMatrix4fv(modeLoc, 1, GL_FALSE, &model[0][0]);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * particle_num, particles.get_particle_position().data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // particle
        particle_shader.use();
        glBindVertexArray(particleVAO);
        unsigned int particle_view = glGetUniformLocation(particle_shader.ID, "view");
        unsigned int particle_proj = glGetUniformLocation(particle_shader.ID, "projection");
        glUniformMatrix4fv(particle_view, 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(particle_proj, 1, GL_FALSE, &projection[0][0]);
        unsigned int particle_model = glGetUniformLocation(particle_shader.ID, "model");
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(-1.0f, 1.5f, -1.0f));
        glUniformMatrix4fv(particle_model, 1, GL_FALSE, &model[0][0]);
        // glDrawArrays(GL_TRIANGLE_FAN, 0, particle_vertices.size());
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, particle_vertices.size(), particle_num);

        // Draw skybox as last
        glDepthFunc(GL_LEQUAL);
        space_box_shader.use();
        view = glm::mat3(glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp)); // Remove translation from the view matrix
        viewLoc = glGetUniformLocation(space_box_shader.ID, "view");
        projLoc = glGetUniformLocation(space_box_shader.ID, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
        // Skybox cube
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LEQUAL);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Measure FPS
        double fps_current_time = glfwGetTime();
        double fps_delta = fps_current_time - fps_last_time;
        frame_num += 1;
        if (fps_delta >= 1.0) {
            int fps = int(double(frame_num) / fps_delta);
            std::stringstream ss;
            ss << window_title.c_str() << " [" << fps << " FPS]";
            glfwSetWindowTitle(window, ss.str().c_str());
            frame_num = 0;
            fps_last_time = fps_current_time;
        }
    }

    glfwTerminate();
    return 0;
}
