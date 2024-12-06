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
    void triggerOptimization(std::vector<std::shared_ptr<VmmSegment>>& segments) {
        // Sort segments by their starting address to facilitate stitching
        std::sort(segments.begin(), segments.end(), [](const std::shared_ptr<VmmSegment>& a, const std::shared_ptr<VmmSegment>& b) {
            return a->segment_ptr < b->segment_ptr;
        });
    
        // Attempt to stitch adjacent segments
        for (size_t i = 0; i < segments.size() - 1; ++i) {
            if (segments[i]->remerge(*segments[i + 1])) {
                // Remove the merged segment
                segments.erase(segments.begin() + i + 1);
                --i; // Adjust index to recheck the current segment with the next one
            }
        }
    
        // Compaction logic: move free blocks to create larger contiguous free spaces
        for (auto& segment : segments) {
            if (segment->free_blocks > 0) {
                size_t freeBlockSize = segment->free_blocks * segment->granul_size;
                auto newSegment = segment->split(freeBlockSize);
                if (newSegment) {
                    segments.push_back(newSegment);
                }
            }
        }
    
        // Sort segments again after compaction
        std::sort(segments.begin(), segments.end(), [](const std::shared_ptr<VmmSegment>& a, const std::shared_ptr<VmmSegment>& b) {
            return a->segment_ptr < b->segment_ptr;
        });
    }

    size_t fragmentationThreshold;
    float fragmentationRatio;
    float memoryUtilization;
    std::vector<size_t> freeBlocks;
    std::vector<size_t> allocatedBlocks;
};
