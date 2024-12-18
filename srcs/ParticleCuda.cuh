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

        void initialize(const std::vector<glm::vec3> &position, const std::vector<glm::vec3> &velocity,
                    const int particle_num, const int threads, const float collision_distance);
        void update_position_velocity(std::vector<glm::vec3> &position, const float mass, const float delta_time);
};

#endif
