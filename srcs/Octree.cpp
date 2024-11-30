#include "Octree.hpp"


Octree::Octree(glm::vec3 min_3d_coord, glm::vec3 max_3d_coord) {
    this->root.min_bound = min_3d_coord;
    this->root.max_bound = max_3d_coord;
    this->root.is_leaf = false;
    subdivide(this->root);
    std::cout << "child " << this->root.children.size() << std::endl;
}

Octree::~Octree() {}

void Octree::insert(std::vector<glm::vec3> position, std::vector<float> mass) {
    for (int i = 0; i < position.size(); i++) {
        // insert_point(this->root, position, mass);
    }
}

// TODO: Add insert function
void Octree::insert_point(Node node, std::vector<glm::vec3> position, std::vector<float> mass) {
    
}

void Octree::subdivide(Node &node) {
    glm::vec3 mid((node.min_bound.x + node.max_bound.x) / 2,
                (node.min_bound.y + node.max_bound.y) / 2,
                (node.min_bound.z + node.max_bound.z) / 2);
    // Bottom
    node.children.push_back(
        create_child_node(
            glm::vec3(node.min_bound.x, node.min_bound.y, mid.z),
            glm::vec3(mid.x, mid.y, node.max_bound.z)
        )
    );
    node.children.push_back(
        create_child_node(
            glm::vec3(node.min_bound.x, node.min_bound.y, node.min_bound.z),
            glm::vec3(mid.x, mid.y, mid.z)
        )
    );
    node.children.push_back(
        create_child_node(
            glm::vec3(mid.x, node.min_bound.y, node.min_bound.z),
            glm::vec3(node.max_bound.x, mid.y, mid.z)
        )
    );
    node.children.push_back(
        create_child_node(
            glm::vec3(mid.x, node.min_bound.y, mid.z),
            glm::vec3(node.max_bound.x, mid.y, node.max_bound.z)
        )
    );
    // Top
    node.children.push_back(
        create_child_node(
            glm::vec3(node.min_bound.x, mid.y, mid.z),
            glm::vec3(mid.x, node.max_bound.y, node.max_bound.z)
        )
    );
    node.children.push_back(
        create_child_node(
            glm::vec3(node.min_bound.x, mid.y, node.min_bound.z),
            glm::vec3(mid.x, node.max_bound.y, mid.z)
        )
    );
    node.children.push_back(
        create_child_node(
            glm::vec3(mid.x, mid.y, node.min_bound.z),
            glm::vec3(node.max_bound.x, node.max_bound.y, mid.z)
        )
    );
    node.children.push_back(
        create_child_node(
            glm::vec3(mid.x, mid.y, mid.z),
            glm::vec3(node.max_bound.x, node.max_bound.y, node.max_bound.z)
        )
    );
    std::cout << "node " << node.children.size() << std::endl;
}

Node Octree::create_child_node(glm::vec3 min_bound, glm::vec3 max_bound) {
    Node new_node;
    new_node.min_bound = min_bound;
    new_node.max_bound = max_bound;

    return new_node;
}
