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

#include "Octree.hpp"
#include "ParticleCuda.cuh"


class Particle {
    private:
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> velocity;
        float mass;
        // glm::vec3 max_3d_coord;
        // glm::vec3 min_3d_coord;
        float collision_distance;
        ParticleCuda particle_cuda;

    public:
        Particle(glm::vec3 center_pos_1, glm::vec3 center_pos_2, float planet_radius, int particle_num_1,
            int particle_num_2, glm::vec3 initial_velocity_1, glm::vec3 initial_velocity_2, float mass, float particle_radius);
        ~Particle();

        std::vector<glm::vec3> get_particle_position();

        void initialize_position(glm::vec3 center_pos, float planet_radius, int particle_num);
        void update_particle(float delta_time);
        // void update_min_max_position(glm::vec3 pos);
        // void reset_min_max_position();
};

#endif
