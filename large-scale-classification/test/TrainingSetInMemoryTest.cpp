#include <gtest/gtest.h>
#include <memory>

#include "TrainingSetInMemory.hpp"

using namespace GBDT;

class TrainingSetInMemoryTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    ptr = std::make_shared<TrainingSetInMemory>(config::TRAINING_PATH ,201);
  }
  static std::shared_ptr<TrainingSet> ptr;
};

shared_ptr<TrainingSet> TrainingSetInMemoryTest::ptr = nullptr;

TEST_F(TrainingSetInMemoryTest, ReadTest) {
  EXPECT_EQ(1, ptr->getLable(0));
  EXPECT_EQ(1, ptr->getLable(1));
  EXPECT_EQ(0.14285714285714285, ptr->getFeature(0, 98));
  EXPECT_EQ(2768967347.76, ptr->getFeature(1, 110));
  EXPECT_EQ(0, ptr->getFeature(1, 200));
}

TEST_F(TrainingSetInMemoryTest, SortTest) {
  for (int i = 0; i < ptr->getSetSize(); i++) {
    std::cout << "Sorting: " << i << std::endl;
    vector<size_type> all(ptr->getFeatureSize());
    std::iota(all.begin(), all.end(), 0);
    ptr->sortSetByFeature(i, all);
  }
}
