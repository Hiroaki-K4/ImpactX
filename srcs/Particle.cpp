#include "Particle.hpp"


Particle::Particle(glm::vec3 center_pos_1, glm::vec3 center_pos_2, float planet_radius,
                    int particle_num_1, int particle_num_2, glm::vec3 initial_velocity_1,
                    glm::vec3 initial_velocity_2, float mass, float particle_radius) {
    this->mass = mass;
    this->collision_distance = particle_radius * 2;
    // reset_min_max_position();
    initialize_position(center_pos_1, planet_radius, particle_num_1);
    for (int i = 0; i < particle_num_1; i++) {
        this->velocity.push_back(initial_velocity_1);
    }
    initialize_position(center_pos_2, planet_radius, particle_num_2);
    for (int i = 0; i < particle_num_2; i++) {
        this->velocity.push_back(initial_velocity_2);
    }
    this->particle_cuda.initialize(this->position, this->velocity, particle_num_1 + particle_num_2,
                                    256, this->collision_distance);
}

Particle::~Particle() {
}

std::vector<glm::vec3> Particle::get_particle_position() {
    return this->position;
}

void Particle::initialize_position(
    glm::vec3 center_pos, float planet_radius, int particle_num) {
    std::random_device rd;   // Seed for the random number engine
    std::mt19937 gen(rd());  // Mersenne Twister engine

    std::uniform_real_distribution<float> angle_phi_dis(-M_PI / 2.0f, M_PI / 2.0f);
    std::uniform_real_distribution<float> angle_theta_dis(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dis(0.0f, planet_radius);
    for (int i = 0; i < particle_num; i++) {
        glm::vec3 pos;
        float angle_phi = angle_phi_dis(gen);
        float angle_theta = angle_theta_dis(gen);
        float radius = radius_dis(gen);

        pos.x = center_pos.x + radius * cos(angle_phi) * cos(angle_theta);
        pos.y = center_pos.y + radius * sin(angle_phi);
        pos.z = center_pos.z + radius * cos(angle_phi) * sin(angle_theta);
        this->position.push_back(pos);
    }
}

void Particle::update_particle(float delta_time) {
    this->particle_cuda.update_position_velocity(this->position, this->mass, delta_time);
}

// void Particle::update_min_max_position(glm::vec3 pos) {
//     this->max_3d_coord.x = std::max(pos.x, this->max_3d_coord.x);
//     this->max_3d_coord.y = std::max(pos.y, this->max_3d_coord.y);
//     this->max_3d_coord.z = std::max(pos.z, this->max_3d_coord.z);
//     this->min_3d_coord.x = std::min(pos.x, this->min_3d_coord.x);
//     this->min_3d_coord.y = std::min(pos.y, this->min_3d_coord.y);
//     this->min_3d_coord.z = std::min(pos.z, this->min_3d_coord.z);
// }

// void Particle::reset_min_max_position() {
//     this->max_3d_coord = glm::vec3(std::numeric_limits<float>::min(),
//                                     std::numeric_limits<float>::min(),
//                                     std::numeric_limits<float>::min());
//     this->min_3d_coord = glm::vec3(std::numeric_limits<float>::max(),
//                                     std::numeric_limits<float>::max(),
//                                     std::numeric_limits<float>::max());
// }
