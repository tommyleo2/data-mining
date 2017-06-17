#include "TrainingSetInMemory.hpp"

#include <algorithm>

using namespace GBDT;

TrainingSetInMemory::TrainingSetInMemory(const string &file_path,
                                         size_type feature_size) :
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
      index_type index = 0;
      index_type expected = 0;
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

double TrainingSetInMemory::getFeature(index_type id, index_type index) const {
  // auto &row = m_data[id];
  // auto it = row.find(index);
  // if (it == row.end()) {
  //   return 0;
  // }
  // return it->second;
  return m_data[id][index];
}

const TrainingSet::TrainingSetRow_t &TrainingSetInMemory::getCase(index_type id) {
  return m_data[id];
}

double &TrainingSetInMemory::getLable(index_type id) {
  return m_label[id];
}

double TrainingSetInMemory::getLable(index_type id) const {
  return m_label[id];
}

size_type TrainingSetInMemory::getSetSize() const {
  return m_label.size();
}
size_type TrainingSetInMemory::getFeatureSize() const {
  return m_feature_size;
}

void TrainingSetInMemory::sortSetByFeature(index_type index, vector<index_type> &ids) {
  std::sort(ids.begin(), ids.end(), [index, this](index_type a, index_type b) {
      return m_data[a][index] > m_data[b][index];
    });
}
