#ifndef TRAININGSET_H
#define TRAININGSET_H

#include "defines.hpp"

namespace GBDT {

  class TrainingSet {
  public:
    using TrainingSetRow_t = vector<double>;
    using TrainingSetCol_t = vector<double>;

    TrainingSet(const string &file_path) : m_file_path(file_path) { }
    virtual double getFeature(index_type id, index_type index) const = 0;

    virtual const TrainingSetRow_t &getCase(index_type id) = 0;

    virtual double getLable(index_type id) const = 0;
    virtual size_type getSetSize() const = 0;
    virtual size_type getFeatureSize() const = 0;

    virtual const vector<index_type> &sortSetByFeature(index_type index, vector<index_type> &ids) = 0;
    //  the sort the entire set, may use cache
    virtual const vector<index_type> &sortSetByFeature(index_type index) = 0;
  protected:
    string m_file_path;
  };

}

#endif /* TRAININGSET_H */
