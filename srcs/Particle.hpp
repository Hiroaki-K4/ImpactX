#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <limits>
#include <cmath>

#include "ParticleCuda.cuh"
#include "ParticleColor.hpp"


class Particle {
    private:
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> velocity;
        std::vector<glm::vec3> color;
        float mass;
        float collision_distance;
        ParticleCuda particle_cuda;
        ParticleColor particle_color;

    public:
        Particle(const glm::vec3 &center_pos_1, const glm::vec3 &center_pos_2, const float planet_radius,
                const int particle_num_1, const int particle_num_2, const glm::vec3 &initial_velocity_1,
                const glm::vec3 &initial_velocity_2, const float mass, const float particle_radius, const int threads);
        ~Particle();

        std::vector<glm::vec3> get_particle_position();
        std::vector<glm::vec3> get_particle_color();

        void initialize(const glm::vec3 &center_pos, const float planet_radius, const int particle_num);
        void update_particle(const float delta_time);
};

#endif
