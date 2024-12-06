#include <memory>
#include <vector>
#include <algorithm>

class FragmentationMonitor {
public:
    FragmentationMonitor(size_t threshold) : fragmentationThreshold(threshold) {}

    void trackMemoryState(const std::vector<std::shared_ptr<VmmSegment>>& segments) {
        freeBlocks.clear();
        allocatedBlocks.clear();

        for (const auto& segment : segments) {
            for (const auto& phyBlock : segment->phy_blocks) {
                if (phyBlock->free) {
                    freeBlocks.push_back(phyBlock->block_size);
                } else {
                    allocatedBlocks.push_back(phyBlock->block_size);
                }
            }
        }
    }

    void evaluateFragmentation() {
        size_t totalFreeMemory = std::accumulate(freeBlocks.begin(), freeBlocks.end(), 0);
        size_t largestContiguousFreeBlock = *std::max_element(freeBlocks.begin(), freeBlocks.end());

        fragmentationRatio = (totalFreeMemory - largestContiguousFreeBlock) / static_cast<float>(totalFreeMemory);
        memoryUtilization = static_cast<float>(allocatedBlocks.size()) / (allocatedBlocks.size() + freeBlocks.size());

        if (fragmentationRatio > fragmentationThreshold) {
            triggerOptimization();
        }
    }

private:
    void triggerOptimization() {
        // Implement stitching and compaction logic here
        // For example, you can call segment->remerge() or segment->split() methods
    }

    size_t fragmentationThreshold;
    float fragmentationRatio;
    float memoryUtilization;
    std::vector<size_t> freeBlocks;
    std::vector<size_t> allocatedBlocks;
};
