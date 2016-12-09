#ifndef CHOWLIUTREE_H
#define CHOWLIUTREE_H

#include <string>
#include "FabMapVocabluary.h"

namespace pyof2
{

class ChowLiuTree
{
public:
    ChowLiuTree(std::shared_ptr<FabMapVocabluary> vocabluary, boost::python::dict settings);
    ChowLiuTree(std::shared_ptr<FabMapVocabluary> vocabluary, cv::Mat chowLiuTree, cv::Mat fabmapTrainData, boost::python::dict settings);
    virtual ~ChowLiuTree();
    
    // These function are exposed to python
    bool addTrainingImage(std::string imagePath);
    void buildChowLiuTree();
    
    void save(std::string filename) const;
    static std::shared_ptr<ChowLiuTree> load(boost::python::dict settings, std::string filename);
    
    bool isTreeBuilt() const;
    std::shared_ptr<FabMapVocabluary> getVocabluary() const;
    cv::Mat getChowLiuTree() const;
    cv::Mat getTrainingData() const;
    
private:
    std::shared_ptr<FabMapVocabluary> vocabluary;
    cv::Mat chowLiuTree;
    cv::Mat fabmapTrainData;
    double lowerInformationBound;
    bool treeBuilt;
};

}

#endif // CHOWLIUTREE_H
