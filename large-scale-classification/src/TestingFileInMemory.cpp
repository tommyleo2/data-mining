#include "TestingSetInMemory.hpp"

#include <fstream>
#include <sstream>
#include <limits>

using namespace GBDT;

TestingSetInMemory::TestingSetInMemory(const string &in_file_path,
                                       const string &out_file_path,
                                       size_t feature_size) :
  TestingSet(in_file_path, out_file_path), m_feature_size(feature_size) {
  LOG_INFO("Start to read test from file...");

  std::fstream file(m_in_file_path, std::fstream::in);
  if (!file) {
    LOG_ERROR("Cannot open file: " << m_in_file_path);
    throw std::runtime_error("Open testing file failed");
  }

  string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    double tmp;
    int index = 0;
    int expected = 0;
    vector<double> row;
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
    m_cases.push_back(std::move(row));
    if (m_cases.size() % 10000 == 0) {
      LOG_INFO("Read: " << m_cases.size());
    }
  }
  LOG_INFO("Reading samples, done!");
}

double &TestingSetInMemory::getFeature(int id, int index) {
  return m_cases[id][index];
}

double TestingSetInMemory::getFeature(int id, int index) const {
  return m_cases[id][index];
}

size_t TestingSetInMemory::getSetSize() const {
  return m_cases.size();
}

size_t TestingSetInMemory::getFeatureSize() const {
  return m_feature_size;
}

double &TestingSetInMemory::getLable(int id) {
  return m_label[id];
}

double TestingSetInMemory::getLable(int id) const {
  return m_label[id];
}

void TestingSetInMemory::dumpResult() const {
  std::fstream out(m_out_file_path, std::fstream::out);
  if (!out) {
    LOG_ERROR("Cannot open file: " << m_out_file_path);
    throw std::runtime_error("Open predicting file failed");
  }

  out.precision(std::numeric_limits<double>::digits10);
  out << std::fixed;
  out << "id,label" << std::endl;
  for (size_t i = 0; i < m_label.size(); i++) {
    out << i << ',' << m_label[i] << std::endl;
  }
}
