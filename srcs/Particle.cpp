#include "Particle.hpp"


Particle::Particle(glm::vec3 center_pos, float planet_radius, int particle_num,
    glm::vec3 velocity, float mass, float particle_radius) {
    this->mass = mass;
    this->collision_distance = particle_radius * 2;
    // reset_min_max_position();
    std::cout << "pre " << particle_num << std::endl;
    initialize_position(center_pos, planet_radius, particle_num);
    std::cout << "pre2 " << particle_num << std::endl;
    for (int i = 0; i < particle_num; i++) {
        this->velocity.push_back(velocity);
    }
    this->particle_cuda = ParticleCuda(this->position, this->velocity, particle_num, 256, this->collision_distance);
}

Particle::~Particle() {
}

std::vector<glm::vec3> Particle::get_particle_position() {
    return this->position;
}

void Particle::initialize_position(
    glm::vec3 center_pos, float planet_radius, int particle_num) {
    std::cout << particle_num << std::endl;
    std::random_device rd;   // Seed for the random number engine
    std::mt19937 gen(rd());  // Mersenne Twister engine

    // Define a distribution between -1 and 1
    std::uniform_real_distribution<float> angle_phi_dis(-M_PI / 2.0f, M_PI / 2.0f);
    std::uniform_real_distribution<float> angle_theta_dis(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dis(0.0f, planet_radius);
    int i = 0;
    while (i < particle_num) {
        glm::vec3 pos;
        float angle_phi = angle_phi_dis(gen);
        float angle_theta = angle_theta_dis(gen);
        float radius = radius_dis(gen);

        pos.x = radius * cos(angle_phi) * cos(angle_theta);
        pos.y = radius * sin(angle_phi);
        pos.z = radius * cos(angle_phi) * sin(angle_theta);

        bool is_distance_valid = true;
        for (int j = 0; j < this->position.size(); j++) {
            float dist = glm::distance(pos, this->position[j]);
            if (dist <= this->collision_distance) {
                is_distance_valid = false;
                break;
            }
        }

        if (is_distance_valid) {
            this->position.push_back(pos);
            // update_min_max_position(pos);
            i++;
        }
    }
}

// void Particle::update_particle(float delta_time) {
//     for (int i = 0; i < this->position.size(); i++) {
//         for (int j = 0; j < this->position.size(); j++) {
//             if (i == j) {
//                 continue;
//             }
//             float dist = glm::distance(this->position[i], this->position[j]);
//             if (dist <= this->collision_distance) {
//                 // Calculate collision
//                 glm::vec3 new_velocity_vec = this->position[i] - this->position[j];
//                 this->velocity[i] = (this->mass - this->mass) / (this->mass + this->mass) * this->velocity[i] +
//                                     2 * this->mass / (this->mass + this->mass) * this->velocity[j];
//             } else {
//                 // Calculate gravity
//                 float accel_power = this->mass * this->mass / std::pow(dist, 2);
//                 glm::vec3 accel = (this->position[j] - this->position[i]) / dist;
//                 accel *= accel_power;
//                 this->velocity[i] += accel * delta_time;
//             }
//         }
//         this->position[i] += this->velocity[i] * delta_time;
//     }
// }

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
