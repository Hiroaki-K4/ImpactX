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
#include <cuda_runtime.h>

#include "Octree.hpp"


class Particle {
    private:
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> velocity;
        std::vector<float> mass;
        // glm::vec3 max_3d_coord;
        // glm::vec3 min_3d_coord;
        float collision_distance;

    public:
        Particle(glm::vec3 center_pos, float planet_radius, int particle_num, glm::vec3 velocity,
            float mass, float particle_radius);
        ~Particle();

        std::vector<glm::vec3> get_particle_position();

        void initialize_position(glm::vec3 center_pos, float planet_radius, int particle_num, float mass);
        void update_particle(float delta_time);
        // void update_min_max_position(glm::vec3 pos);
        // void reset_min_max_position();
};

#endif
