#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

__global__ void update_particle_kernel(glm::vec3 *cu_position, glm::vec3 *cu_velocity, const float mass,
    const float delta_time, const int num_particles, const float collision_distance);

#endif
