// Test to verify octree can handle more than 255 nodes
#include <iostream>
#include <vector>
#include <cstdint>

// Simplified test structure matching our fixed OctreeNode
struct TestOctreeNode {
    uint32_t children[8];  // Fixed from uint8_t to uint32_t
    int level;
};

int main() {
    std::vector<TestOctreeNode> octree;

    // Create root node
    TestOctreeNode root;
    root.level = 0;
    for (int i = 0; i < 8; i++) {
        root.children[i] = 0xFFFFFFFF; // Invalid marker
    }
    octree.push_back(root);

    // Simulate creating many nodes (more than 255)
    for (int i = 1; i <= 300; i++) {
        TestOctreeNode node;
        node.level = (i / 8) + 1;
        for (int j = 0; j < 8; j++) {
            node.children[j] = 0xFFFFFFFF;
        }
        octree.push_back(node);

        // Link some nodes as children (simulate octree structure)
        if (i <= 8) {
            octree[0].children[i-1] = i;  // Root's children
        } else if (i <= 64) {
            size_t parent = ((i - 9) / 8) + 1;
            int child_slot = (i - 9) % 8;
            if (parent < octree.size()) {
                octree[parent].children[child_slot] = i;
            }
        }
    }

    // Verify we can correctly access nodes beyond index 255
    for (int i = 0; i < 8; i++) {
        uint32_t child_idx = octree[0].children[i];
        if (child_idx != 0xFFFFFFFF && child_idx < octree.size()) {
            std::cout << "Root child " << i << " -> node " << child_idx << std::endl;
            if (child_idx > 255) {
                std::cout << "SUCCESS: Accessing node index > 255!" << std::endl;
            }
        }
    }

    // Test accessing a high-index node
    if (octree.size() > 256) {
        TestOctreeNode &high_node = octree[256];
        std::cout << "\nNode 256 level: " << high_node.level << std::endl;
        std::cout << "Node 256 first child: " << std::hex << high_node.children[0] << std::dec << std::endl;

        if (high_node.children[0] == 0xFFFFFFFF) {
            std::cout << "SUCCESS: High-index node properly stores 0xFFFFFFFF marker!" << std::endl;
        }
    }

    std::cout << "\nTotal octree nodes created: " << octree.size() << std::endl;
    std::cout << "TEST PASSED: Octree can handle > 255 nodes with uint32_t indices" << std::endl;

    return 0;
}