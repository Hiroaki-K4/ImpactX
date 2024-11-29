#include "Octree.hpp"

Octree::Octree(glm::vec3 max_3d_coord, glm::vec3 min_3d_coord) {
    this->root.x_min_max = std::make_pair(min_3d_coord.x, max_3d_coord.x);
    this->root.y_min_max = std::make_pair(min_3d_coord.y, max_3d_coord.y);
    this->root.z_min_max = std::make_pair(min_3d_coord.z, max_3d_coord.z);
    this->root.is_empty = false;
}

Octree::~Octree() {}

void Octree::insert(std::vector<glm::vec3> position, std::vector<float> mass) {
    for (int i = 0; i < position.size(); i++) {
        // insert_point(this->root, position, mass);
    }
}

// TODO: Fix insert_point function
void Octree::insert_point(Node node, std::vector<glm::vec3> position, std::vector<float> mass) {

}

void Octree::subdivide() {}
