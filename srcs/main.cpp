#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.hpp"

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

    float planeVertices[] = {
        10.0f, -0.5f,  10.0f,  2.0f, 0.0f,
        -10.0f, -0.5f,  10.0f,  0.0f, 0.0f,
        -10.0f, -0.5f, -10.0f,  0.0f, 2.0f,

        10.0f, -0.5f,  10.0f,  2.0f, 0.0f,
        -10.0f, -0.5f, -10.0f,  0.0f, 2.0f,
        10.0f, -0.5f, -10.0f,  2.0f, 2.0f								
    };

    // Initialize window
    glViewport(0, 0, window_w, window_h);
    glClearColor(0.4f, 0.7f, 0.9f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // cube VAO
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

    // plane VAO;
    unsigned int planeVAO, planeVBO;
    glGenVertexArrays(1, &planeVAO);
    glGenBuffers(1, &planeVBO);
    glBindVertexArray(planeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), &planeVertices, GL_STATIC_DRAW);
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    unsigned int cubeTexture = loadTexture("../textures/awesomeface.png", true);
    unsigned int floor_texture = loadTexture("../textures/floor.png", true);

    double last_time = glfwGetTime();
    double fps_last_time = glfwGetTime();
    int frame_num = 0;
    shader.use();
    shader.setInt("texture1", 0);
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        // glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov), float(window_w) / float(window_h), 0.1f, 100.0f);
        unsigned int viewLoc = glGetUniformLocation(shader.ID, "view");
        unsigned int projLoc = glGetUniformLocation(shader.ID, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);

        // cubes
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
        // floor
        glBindVertexArray(planeVAO);
        glBindTexture(GL_TEXTURE_2D, floor_texture);
        model = glm::mat4(1.0f);
        // unsigned int modeLoc = glGetUniformLocation(shader.ID, "model");
        glUniformMatrix4fv(modeLoc, 1, GL_FALSE, &model[0][0]);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

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
