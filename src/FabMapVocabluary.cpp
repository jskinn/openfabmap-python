#ifdef OPENCV2P4
//#include <opencv2/nonfree/nonfree.hpp>
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <bowmsctrainer.hpp>
#include "FabMapVocabluary.h"
#include "detectorsAndExtractors.h"

// ----------------- FabMapVocabluary -----------------

pyof2::FabMapVocabluary::FabMapVocabluary(cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> extractor, cv::Mat vocabluary) :
        detector(std::move(detector)),
        extractor(std::move(extractor)),
        vocab(std::move(vocabluary))
{
    
}

pyof2::FabMapVocabluary::~FabMapVocabluary()
{
    
}
    
cv::Mat pyof2::FabMapVocabluary::getVocabluary() const
{
    return vocab;
}

cv::Mat pyof2::FabMapVocabluary::generateBOWImageDescs(const cv::Mat& frame) const
{
    //use a FLANN matcher to generate bag-of-words representations
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    cv::BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);
    
    cv::Mat bow;
    std::vector<cv::KeyPoint> kpts;

    detector->detect(frame, kpts);
    bide.compute(frame, kpts, bow);
    return bow;
}

void pyof2::FabMapVocabluary::save(cv::FileStorage fileStorage) const
{
    fileStorage << "Vocabluary" << vocab;
    //TODO: Find a way to save the detector and extractor settings.
}

std::shared_ptr<pyof2::FabMapVocabluary> pyof2::FabMapVocabluary::load(cv::FileStorage fileStorage)
{
    cv::Mat vocab;
    fileStorage["Vocabluary"] >> vocab;
    
    // TODO: Find a way to save the settings used to make the detector and extractor
    
    return std::make_shared<pyof2::FabMapVocabluary>(
        pyof2::generateDetector(boost::python::dict()),
        pyof2::generateExtractor(boost::python::dict()),
        vocab);
}

// ----------------- FabMapVocabluaryBuilder -----------------

pyof2::FabMapVocabluaryBuilder::FabMapVocabluaryBuilder(boost::python::dict settings) :
        detector(pyof2::generateDetector(settings)),
        extractor(pyof2::generateExtractor(settings)),
        vocabTrainData(),
        clusterRadius(0.45)
{
    if (settings.has_key("VocabTrainOptions"))
    {
        boost::python::dict trainSettings = boost::python::extract<boost::python::dict>(settings.get("VocabTrainOptions"));
        if (trainSettings.has_key("ClusterSize"))
        {
            clusterRadius = boost::python::extract<double>(trainSettings.get("ClusterSize"));
        }
    }
}

pyof2::FabMapVocabluaryBuilder::~FabMapVocabluaryBuilder()
{
    
}

bool pyof2::FabMapVocabluaryBuilder::addTrainingImage(std::string imagePath)
{
    cv::Mat descs, feats;
    std::vector<cv::KeyPoint> kpts;
    
    cv::Mat frame = cv::imread(imagePath, CV_LOAD_IMAGE_UNCHANGED);
    if (frame.data)
    {
        //detect & extract features
        detector->detect(frame, kpts);
        extractor->compute(frame, kpts, descs);

        //add all descriptors to the training data 
        vocabTrainData.push_back(descs);
        return true;
    }
    return false;
}

std::shared_ptr<pyof2::FabMapVocabluary> pyof2::FabMapVocabluaryBuilder::buildVocabluary()
{
    // Build the vocab
    of2::BOWMSCTrainer trainer(clusterRadius);
    trainer.add(vocabTrainData);
    cv::Mat vocab = trainer.cluster();
    
    // Return the vocab object
    return std::make_shared<pyof2::FabMapVocabluary>(detector, extractor, std::move(vocab));
}

