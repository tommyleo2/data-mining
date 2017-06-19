#ifndef CONFIG_H
#define CONFIG_H

namespace GBDT {
  namespace config {
    const char * const TRAINING_PATH = "./data/train_data.txt";
    const char * const TRAINING_PATH_TEST = "./data/train_data_test.txt";
    const char * const TEST_PATH = "./data/test_data.txt";
    const char * const OUTPUT_PATH = "./data/predict.csv";

    const double ALPHA = 0.1;    //  shrinkage
    const double LAMBDA = 0.01;  //  L2 regularization term on weights
    const double GAMMA = 1.0;    //  minimum loss reduction required to make a further partition

    const unsigned int MAX_TREE_DEPTH = 6;
    const unsigned int ITERATION_TIMES = 10;
  }
}

#endif /* CONFIG_H */
