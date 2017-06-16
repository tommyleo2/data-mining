#include <gtest/gtest.h>
#include <memory>

#include "TrainingSetInMemory.hpp"

using namespace GBDT;

TEST(TrainingSetInMemoryTest, ReadTest) {
  std::shared_ptr<TrainingSet> ptr =
    std::make_shared<TrainingSetInMemory>(config::TRAINING_PATH_TEST ,201);
  EXPECT_EQ(1, ptr->getLable(0));
  EXPECT_EQ(1, ptr->getLable(1));
  EXPECT_EQ(0.14285714285714285, ptr->getFeature(0, 98));
  EXPECT_EQ(2768967347.76, ptr->getFeature(1, 110));
  EXPECT_EQ(0, ptr->getFeature(1, 200));
}
