#ifndef CONFIG_H
#define CONFIG_H

namespace GBDT {
  namespace config {
    const char * const TRAINING_PATH = "./data/train_data.txt";
    const char * const TRAINING_PATH_TEST = "./data/train_data_test.txt";
    const char * const TEST_PATH = "./data/test_data.txt";
    const char * const OUTPUT_PATH = "./data/predict.csv";

    const double SHRINKAGE = 0.1;
    const double REGULARIZATION = 0;
    const double LAMBDA = 1;

    const unsigned int MAX_TREE_DEPTH = 6;
    const unsigned int ITERATION_TIMES = 3000;
  }
}

#endif /* CONFIG_H */
