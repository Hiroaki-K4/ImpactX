#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// __global__ void update_particle_kernel(glm::vec2 *cu_position, glm::vec2 *cu_velocity,
//                                        glm::vec3 *cu_color, glm::vec2 gravity_pos, float delta_time,
//                                        float aspect_ratio, int num_particles, float max_distance, float gravity_strength);

__global__ void update_particle_kernel(glm::vec3 *cu_position, glm::vec3 *cu_velocity, float mass,
                                        float delta_time, int num_particles, float collision_distance);

#endif
