# Large Scale Classification with GBDT

A GBDT algorithm implementation to solve large scale classification problem

## Quick Start

1. `mkdir build && cd build`
2. `cmake ..` or `cmake -DBUILD_TESTING=OFF ..` if you don't have GTest or don't want to test
3. `cd .. && ./bin/GBDT`

## Parameter Tuning

The following parameters are provided and configurable
Details in [config.hpp](https://github.com/tommyleo2/data-mining/blob/master/large-scale-classification/include/config.hpp)

1. ETA：shrinkage, every prediction by a single tree is multiplied by this value.
2. LAMBDA：L2 regularization
3. GAMMA：regularization, threshold of split gain
4. MAX_TREE_DEPTH：max tree depth
5. ITERATION_TIMES：iteration times, number of trees to learn
6. THREAD_NUM：thread number in thread pool

## Extend

You may want to have your own implementation on some of the modules. Just subclass the classes in framework and load you own classes in main
For more usage checkout [main.cpp](https://github.com/tommyleo2/data-mining/blob/master/large-scale-classification/src/main.cpp)
