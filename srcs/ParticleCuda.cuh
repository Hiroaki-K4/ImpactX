#ifndef PARTICLECUDA_CUH
#define PARTICLECUDA_CUH

#include <cuda_runtime.h>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>

#include "kernel.cuh"


class ParticleCuda {
    private:
        glm::vec3 *cu_position;
        glm::vec3 *cu_velocity;
        int threads;
        int blocks;
        float collision_distance;

    public:
        ParticleCuda();
        ~ParticleCuda();

        void initialize(std::vector<glm::vec3> &position, std::vector<glm::vec3> &velocity,
                    int particle_num, int threads, float collision_distance);
        void update_position_velocity(std::vector<glm::vec3> &position, float mass, float delta_time);
};

#endif
