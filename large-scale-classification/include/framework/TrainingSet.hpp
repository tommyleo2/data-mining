#ifndef TRAININGSET_H
#define TRAININGSET_H

#include "defines.hpp"

namespace GBDT {

  class TrainingSet {
  public:
    using TrainingSetRow_t = vector<double>;
    using TrainingSetCol_t = vector<double>;

    TrainingSet(const string &file_path) : m_file_path(file_path) { }
    virtual double &getFeature(int id, int index) = 0;
    virtual double getFeature(int id, int index) const = 0;

    virtual const TrainingSetRow_t &getCase(int id) = 0;

    virtual double &getLable(int id) = 0;
    virtual double getLable(int id) const = 0;
    virtual size_t getSetSize() const = 0;
    virtual size_t getFeatureSize() const = 0;

    virtual void sortSetByFeature(int index, vector<size_t> &ids) = 0;
  protected:
    string m_file_path;
  };

}

#endif /* TRAININGSET_H */
