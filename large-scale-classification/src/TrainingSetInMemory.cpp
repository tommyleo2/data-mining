#include "TrainingSetInMemory.hpp"

using namespace GBDT;

TrainingSetInMemory::TrainingSetInMemory(size_t feature_size) :
  m_feature_size(feature_size) {
  std::fstream file(config::TRAINING_PATH, std::fstream::in);
  string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    double tmp;
    ss >> tmp;
    m_label.push_back(tmp);
    int index = 0;
    int expected = 0;
    TrainingSetRow_t row;
    while (ss >> index) {
      index--;
      ss.ignore();
      while (index > expected) {
        row.push_back(0);
        expected++;
      }
      ss >> tmp;
      row.push_back(tmp);
      expected++;
    }
    while (expected < m_feature_size) {
      row.push_back(0);
      expected++;
    }
    m_data.push_back(std::move(row));
  }
}


double &TrainingSetInMemory::getFeature(int id, int index) {
  return m_data[id][index];
}

double TrainingSetInMemory::getFeature(int id, int index) const {
  return m_data[id][index];
}

double &TrainingSetInMemory::getLable(int id) {
  return m_label[id];
}

double TrainingSetInMemory::getLable(int id) const {
  return m_label[id];
}

size_t TrainingSetInMemory::getSetSize() const {
  return m_label.size();
}
size_t TrainingSetInMemory::getFeatureSize() const {
  return m_feature_size;
}
