#ifndef CONFIG_H
#define CONFIG_H

namespace Config {
  namespace Path {
    const char * const TRAINING_SET = "data/training_data/save_train.csv";
    const char * const TEST_SET =  "data/predict_data/save_test.csv";
    const char * const PREDICT_SET =  "data/predict_result/result.csv";
    const char * const TRAINED_VARIABLES = "data/trained_variables.csv";
  }
  namespace Params {
    const double ALPHA = 0.1;
    const double ALPHA_STEP = 1.1;
  }
}

#endif /* CONFIG_H */
