#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include "config.hpp"

class LinearRegression {
  using value_type = double;
public:
  LinearRegression();
  ~LinearRegression();

  void learn(unsigned long long times);
  void predict();

private:
  std::vector<value_type> variables;
  std::vector< std::vector<value_type> > training_data;

  double alpha;

private:
  void setup();
  void loadTrainingFile();
  void iterate();

  void saveVariables();
  int getVariableNum();

private:
  std::vector<value_type> current_diff;
  value_type previous_cost;

private:
  void calculateDiff(const std::vector<value_type> &derivatives,
                     std::vector<value_type> &diff_vector);
};


#endif /* LINEARREGRESSION_H */
