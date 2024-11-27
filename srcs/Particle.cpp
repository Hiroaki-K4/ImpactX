#include "Particle.hpp"

Particle::Particle(glm::vec3 center_pos, float planet_radius, int particle_num, glm::vec3 velocity) {
    initialize_position(center_pos, planet_radius, particle_num);
    for (int i = 0; i < particle_num; i++) {
        this->velocity.push_back(velocity);
    }
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

    // Define a distribution between -1 and 1
    std::uniform_real_distribution<float> angle_phi_dis(-M_PI / 2.0f, M_PI / 2.0f);
    std::uniform_real_distribution<float> angle_theta_dis(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dis(0.0f, planet_radius);
    for (int i = 0; i < particle_num; i++) {
        glm::vec3 pos;
        float angle_phi = angle_phi_dis(gen);
        float angle_theta = angle_theta_dis(gen);
        float radius = radius_dis(gen);

        pos.x = radius * cos(angle_phi) * cos(angle_theta);
        pos.y = radius * sin(angle_phi);
        pos.z = radius * cos(angle_phi) * sin(angle_theta);
        this->position.push_back(pos);
        // std::cout << "pos: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
    }
}

void Particle::update_position(float delta_time) {
    for (int i = 0; i < this->position.size(); i++) {
        for (int j = 0; j < this->position.size(); j++) {
            continue;
        }
        this->position[i] += this->velocity[i] * delta_time;

    }
}
