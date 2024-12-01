#include "Octree.hpp"


Octree::Octree(glm::vec3 min_3d_coord, glm::vec3 max_3d_coord) {
    this->root.min_bound = min_3d_coord;
    this->root.max_bound = max_3d_coord;
    this->root.is_leaf = false;
    this->root.is_empty = false;
    subdivide(this->root);
}

Octree::~Octree() {}

bool Octree::is_point_inside(Node &node, glm::vec3 position) {
    if (position.x < node.min_bound.x || position.x > node.max_bound.x ||
        position.y < node.min_bound.y || position.y > node.max_bound.y ||
        position.z < node.min_bound.z || position.z > node.max_bound.z) {
        return false;
    }
    return true;
}

void Octree::insert(std::vector<glm::vec3> &position, std::vector<float> &mass) {
    for (int i = 0; i < position.size(); i++) {
        // TODO: Fix insert function because it is too slow.
        if (i % 1000 == 0) {
            std::cout << "i: " << i << std::endl;
        }
        insert_point(this->root, position[i], mass[i]);
    }
}

void Octree::insert_point(Node &node, glm::vec3 position, float mass) {
    if (!is_point_inside(node, position)) {
        return;
    }

    // is_empty -> add point
    // is_leaf -> subdivide
    // !is_empty -> go down tree
    if (node.is_empty) {
        node.position = position;
        node.mass = mass;
        node.is_empty = false;
    } else {
        if (node.is_leaf) {
            subdivide(node);
        }
        for (auto& child : node.children) {
            insert_point(child, position, mass);
        }
    }
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
}

Node Octree::create_child_node(glm::vec3 min_bound, glm::vec3 max_bound) {
    Node new_node;
    new_node.min_bound = min_bound;
    new_node.max_bound = max_bound;

    return new_node;
}
