#ifndef OCTREE_HPP
#define OCTREE_HPP

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <utility>
#include <vector>


struct Node {
    float mass;
    glm::vec3 position;
    std::pair<float, float> x_min_max;
    std::pair<float, float> y_min_max;
    std::pair<float, float> z_min_max;
    std::vector<Node> children[8];
    Node *prev_node;
};


// TODO: Initialize Octree class
class Octree {
    private:

    public:
        Octree(glm::vec3 max_3d_coord, glm::vec3 min_3d_coord);
        ~Octree();
};

#endif
