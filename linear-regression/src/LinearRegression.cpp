#include "LinearRegression.hpp"

LinearRegression::LinearRegression() : alpha(Config::Params::ALPHA){
  setup();
  current_diff.resize(training_data.size());
}

LinearRegression::~LinearRegression() {
  saveVariables();
}

void LinearRegression::learn(unsigned long long times) {
  loadTrainingFile();
  calculateDiff(variables, current_diff);
  unsigned long long it = 1;
  while (it <= times) {
    for (int i = 0; i < 100; i++) {
      std::cout << "Iteration " << it << ", ";
      iterate();
      std::flush(std::cout);
      it++;
    }
    saveVariables();
  }
}

void LinearRegression::predict() {
  std::vector<value_type> raw_data(variables.size());
  std::ifstream test_file(Config::Path::TEST_SET);
  if (test_file.is_open()) {
    test_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::ofstream predict_file(Config::Path::PREDICT_SET, std::ios::trunc);
    predict_file << "Id,reference" << std::endl;
    std::string buffer;
    while (std::getline(test_file, buffer)) {
      std::stringstream ss;
      ss << buffer;
      int id;
      ss >> id;
      ss.ignore(1);
      raw_data.clear();
      raw_data.push_back(1);
      while (ss) {
        value_type tmp;
        ss >> tmp;
        ss.ignore(1);
        raw_data.push_back(tmp);
      }
      value_type reference = 0;
      for (size_t i = 0; i < raw_data.size(); i++) {
        reference += raw_data[i] * variables[i];
      }
      std::cout << "ID: " << id << ", "
                << "Reference: " << reference << std::endl;
      predict_file << id << "," << reference << std::endl;
    }
  } else {
    std::cerr << "Cannot load test set" << std::endl;
  }
}

void LinearRegression::iterate() {
  static std::vector<value_type> variables_derivative(variables.size());

  static std::vector<value_type> variables_tmp(variables.size());

  //  Solve derivative
  for (size_t i = 0; i < variables.size(); i++) {
    value_type variable_derivative = 0;
    for (size_t j = 0; j < training_data.size(); j++) {
      variable_derivative += current_diff[j] * training_data[j][i];
    }
    variables_derivative[i] = variable_derivative / training_data.size();
  }

  //  Find suitable alpha
  while (true) {
    //  Try to update variables
    for (size_t i = 0; i < variables_tmp.size(); i++) {
      variables_tmp[i] = variables[i] - alpha * variables_derivative[i];
    }

    //  Calculate difference
    calculateDiff(variables_tmp, current_diff);

    //  Calculate cost
    value_type current_cost = 0;
    for (size_t i = 0; i < training_data.size(); i++) {
      current_cost += current_diff[i] * current_diff[i];
    }
    current_cost /= 2 * training_data.size();
    if (current_cost < previous_cost) {
      std::cout << "Current cost: " << current_cost << ", "
                << "Improvement: " << previous_cost - current_cost << ", "
                << "alpha: " << alpha << std::endl;
      previous_cost = current_cost;
      variables = variables_tmp;
      break;
    }
    alpha /= 2;
  }
  alpha *= Config::Params::ALPHA_STEP;
}

void LinearRegression::setup() {
  std::cout << "Setting up" << std::endl;
  std::cout.precision(10);
  std::cout << std::fixed;
  std::ifstream variables_file(Config::Path::TRAINED_VARIABLES);
  if (variables_file.is_open()) {
    //  read value from file (continue learning from previous result)
    std::cout << "Reading variables from previous result" << std::endl;
    variables_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string buffer;
    std::getline(variables_file, buffer);
    std::stringstream ss;
    ss << buffer;
    int i = 0;
    while (std::getline(ss, buffer, ',')) {
      std::cout << "ID: " << i++ << ", "
                << "value: " << buffer << std::endl;
      variables.push_back(std::stod(buffer));
    }
    std::cout << std::endl;
    variables_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    variables_file >> previous_cost;
    std::cout << "Cost: " << previous_cost << std::endl;
  } else {
    //  set default variable value to 1 (initialize learning process)
    std::cout << "Set variables to default value, 1" << std::endl;
    variables.assign(getVariableNum(), 1);
    previous_cost = std::numeric_limits<value_type>::max();
  }
}

void LinearRegression::loadTrainingFile() {
  std::cout << "Reading training set" << std::endl;
  std::ifstream training_file;
  // training_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  training_file.open(Config::Path::TRAINING_SET);
  training_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::string buffer;
  while (std::getline(training_file, buffer)) {
    std::stringstream ss;
    std::vector<value_type> row;
    ss << buffer;
    int id;
    ss >> id;
    ss.ignore(1);
    row.push_back(1);
    while (ss) {
      value_type value;
      ss >> value;
      ss.ignore(1);
      row.push_back(value);
    }
    row.shrink_to_fit();
    training_data.push_back(std::move(row));
    std::cout << "Read " << id << std::endl;
  }
  training_data.shrink_to_fit();
  std::cout << "Reading training set is done" << std::endl;
}

void LinearRegression::saveVariables() {
  std::ofstream variables_file(Config::Path::TRAINED_VARIABLES, std::ios::trunc);
  size_t i = 0;
  for ( ; i < variables.size() - 1; ++i) {
    variables_file << "variable" << i << ',';
  }
  variables_file << "variable" << i << std::endl;
  variables_file.precision(10);
  variables_file << std::fixed;
  for (i = 0; i < variables.size() - 1; i++) {
    variables_file << variables[i] << ',';
  }
  variables_file << variables[i] << std::endl;
  variables_file << "Cost" << std::endl
                 << previous_cost << std::endl;
  variables_file << "Alpha" << std::endl
                 << alpha << std::endl;
}

int LinearRegression::getVariableNum() {
  std::ifstream traning_set_file(Config::Path::TRAINING_SET);
  std::string header;
  std::getline(traning_set_file, header);
  int result = 0;
  for (char ch : header) {
    if (ch == ',') {
      result++;
    }
  }
  return result;
}


void LinearRegression::calculateDiff(const std::vector<value_type> &derivatives,
                                     std::vector<value_type> &diff_vector) {
  diff_vector.clear();
  for (size_t j = 0; j < training_data.size(); j++) {
    diff_vector.push_back(0);
    for (size_t k = 0; k < training_data[j].size() - 1; k++) {
      diff_vector.back() += derivatives[k] * training_data[j][k];
    }
    diff_vector.back() -= training_data[j].back();
  }
}
