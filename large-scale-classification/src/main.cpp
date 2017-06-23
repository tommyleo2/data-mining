#include "GBDTFramework.hpp"
#include "GBDTImpl.hpp"
#include "config.hpp"

using namespace GBDT;
using std::make_shared;

int main(void) {
  shared_ptr<TrainingSet> training_set =
    make_shared<TrainingSetInMemory>(config::TRAINING_PATH_TEST, 201);

  shared_ptr<TestingSet> testing_set =
    make_shared<TestingSetInMemory>(config::TEST_PATH_TEST, config::OUTPUT_PATH, 201);

  shared_ptr<LossFunction> loss_func =
    make_shared<SquareLossFunction>(training_set);

  shared_ptr<DecisionTree> decision_tree =
    make_shared<DecisionTreeInMemory>(training_set, loss_func, config::MODEL_PATH);

  GBDTAlgorithm gbdt(training_set, testing_set, decision_tree);

  gbdt.learn();
  gbdt.predict();

  return 0;
}
