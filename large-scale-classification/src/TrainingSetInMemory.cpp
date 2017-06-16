#include "TrainingSetInMemory.hpp"

#include <algorithm>

using namespace GBDT;

TrainingSetInMemory::TrainingSetInMemory(const string &file_path,
                                         size_t feature_size) :
  TrainingSet(file_path), m_feature_size(feature_size) {

  LOG_INFO("Start to read samples from file...");

  std::fstream file(file_path, std::fstream::in);
  if (!file) {
    LOG_ERROR("Cannot open file: " << file_path);
    throw std::runtime_error("Open training file failed");
  }

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
    if (m_data.size() % 10000 == 0) {
      LOG_INFO("Read: " << m_data.size());
    }
  }
  LOG_INFO("Reading samples, done!");
}


double &TrainingSetInMemory::getFeature(int id, int index) {
  return m_data[id][index];
}

double TrainingSetInMemory::getFeature(int id, int index) const {
  return m_data[id][index];
}

const TrainingSet::TrainingSetRow_t &TrainingSetInMemory::getCase(int id) {
  return m_data[id];
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

void TrainingSetInMemory::sortSetByFeature(int index, vector<size_t> &ids) {
  std::sort(ids.begin(), ids.end(), [index, this](int a, int b) {
      return m_data[a][index] > m_data[b][index];
    });
}
