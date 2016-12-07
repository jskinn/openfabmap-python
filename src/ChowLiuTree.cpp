#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chowliutree.hpp>
#include "ChowLiuTree.h"

// ----------------- ChowLiuTree -----------------

pyof2::ChowLiuTree::ChowLiuTree(std::shared_ptr<FabMapVocabluary> vocabluary, double lowerInformationBound) :
    vocabluary(vocabluary),
    chowLiuTree(),
    fabmapTrainData(),
    lowerInformationBound(lowerInformationBound),
    treeBuilt(false)
{
    
}

pyof2::ChowLiuTree::ChowLiuTree(std::shared_ptr<FabMapVocabluary> vocabluary, cv::Mat chowLiuTree, cv::Mat fabmapTrainData, double lowerInformationBound) :
    vocabluary(vocabluary),
    chowLiuTree(std::move(chowLiuTree)),
    fabmapTrainData(std::move(fabmapTrainData)),
    lowerInformationBound(lowerInformationBound),
    treeBuilt(!this->chowLiuTree.empty())
{
    
}

pyof2::ChowLiuTree::~ChowLiuTree()
{
    
}

bool pyof2::ChowLiuTree::addTrainingImage(std::string imagePath)
{
    cv::Mat frame = cv::imread(imagePath, CV_LOAD_IMAGE_UNCHANGED);
    if (frame.data)
    {
        cv::Mat bow = vocabluary->generateBOWImageDescs(frame);
        fabmapTrainData.push_back(std::move(bow));
        treeBuilt = false;
        return true;
    }
    return false;
}

void pyof2::ChowLiuTree::buildChowLiuTree()
{
    of2::ChowLiuTree tree;
    tree.add(fabmapTrainData);
    chowLiuTree = tree.make(lowerInformationBound);
    treeBuilt = true;
}

bool pyof2::ChowLiuTree::isTreeBuilt() const
{
    return treeBuilt;
}

std::shared_ptr<pyof2::FabMapVocabluary> pyof2::ChowLiuTree::getVocabluary() const
{
    return vocabluary;
}
    
cv::Mat pyof2::ChowLiuTree::getChowLiuTree() const
{
    return chowLiuTree;
}

cv::Mat pyof2::ChowLiuTree::getTrainingData() const
{
    return fabmapTrainData;
}
    
void pyof2::ChowLiuTree::save(std::string filename) const
{
    cv::FileStorage fs;	
    fs.open(filename, cv::FileStorage::WRITE);
    vocabluary->save(fs);
    
    fs << "LowerInformationBound" << lowerInformationBound;
    if (treeBuilt)
    {
        fs << "ChowLiuTree" << chowLiuTree;
        fs << "FabMapTrainingData" << fabmapTrainData;
    }
    
    fs.release();
}

std::shared_ptr<pyof2::ChowLiuTree> pyof2::ChowLiuTree::load(std::string filename)
{
    cv::FileStorage fs;	
    fs.open(filename, cv::FileStorage::READ);
    
    std::shared_ptr<pyof2::FabMapVocabluary> vocab = pyof2::FabMapVocabluary::load(fs);
    
    double lowerInformationBound;
    fs["LowerInformationBound"] >> lowerInformationBound;
    
    cv::Mat chowLiuTree;
    fs["ChowLiuTree"] >> chowLiuTree;
    
    cv::Mat fabmapTrainData;
    fs["FabMapTrainingData"] >> fabmapTrainData;
    
    fs.release();
    
    return std::make_shared<pyof2::ChowLiuTree>(vocab, chowLiuTree, fabmapTrainData, lowerInformationBound);
}
