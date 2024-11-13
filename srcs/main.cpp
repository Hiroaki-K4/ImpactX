#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "Shader.hpp"


void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
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

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Error: Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Initialize window
    glViewport(0, 0, window_w, window_h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    Shader particle_shader("../shaders/particle.vs", "../shaders/particle.fs");

    double last_time = glfwGetTime();
    double fps_last_time = glfwGetTime();
    int frame_num = 0;
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT);

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
