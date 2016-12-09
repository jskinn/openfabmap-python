#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chowliutree.hpp>
#include "ChowLiuTree.h"

// ----------------- ChowLiuTree -----------------

pyof2::ChowLiuTree::ChowLiuTree(std::shared_ptr<FabMapVocabluary> vocabluary, boost::python::dict settings) :
    ChowLiuTree(vocabluary, cv::Mat(), cv::Mat(), settings)
{
    
}

pyof2::ChowLiuTree::ChowLiuTree(std::shared_ptr<FabMapVocabluary> vocabluary, cv::Mat chowLiuTree, cv::Mat fabmapTrainData, boost::python::dict settings) :
    vocabluary(vocabluary),
    chowLiuTree(std::move(chowLiuTree)),
    fabmapTrainData(std::move(fabmapTrainData)),
    lowerInformationBound(0.0005),
    treeBuilt(!this->chowLiuTree.empty())
{
    if (settings.has_key("ChowLiuOptions"))
    {
        boost::python::dict trainSettings = boost::python::extract<boost::python::dict>(settings.get("ChowLiuOptions"));
        if (trainSettings.has_key("LowerInfoBound"))
        {
            lowerInformationBound = boost::python::extract<double>(trainSettings.get("LowerInfoBound"));
        }
    }
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
    if (treeBuilt)
    {
        fs << "ChowLiuTree" << chowLiuTree;
        fs << "FabMapTrainingData" << fabmapTrainData;
    }
    fs.release();
}

std::shared_ptr<pyof2::ChowLiuTree> pyof2::ChowLiuTree::load(boost::python::dict settings, std::string filename)
{
    cv::FileStorage fs;	
    fs.open(filename, cv::FileStorage::READ);
    
    std::shared_ptr<pyof2::FabMapVocabluary> vocab = pyof2::FabMapVocabluary::load(settings, fs);
    
    cv::Mat chowLiuTree;
    fs["ChowLiuTree"] >> chowLiuTree;
    
    cv::Mat fabmapTrainData;
    fs["FabMapTrainingData"] >> fabmapTrainData;
    
    fs.release();
    
    return std::make_shared<pyof2::ChowLiuTree>(vocab, chowLiuTree, fabmapTrainData, settings);
}
