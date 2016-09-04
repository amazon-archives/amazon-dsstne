#include <map>
#include <string>
#include <sstream>
#include <unordered_map>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include "NetCDFhelper.h"

using namespace std;

class TestNetCDFhelper : public CppUnit::TestFixture
{
    const static map<string, unsigned int> validFeatureIndex;

public:
    void TestLoadIndexWithValidInput() {
        // Seed the input stream with valid input
        stringstream inputStream;
        for (const auto &entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain no error messages", 
            outputStream.str().find("Error") == string::npos);
        CPPUNIT_ASSERT_MESSAGE("Number of entries in feature index should be equal to number of lines of input",
            validFeatureIndex.size() == labelsToIndices.size());

        // Check that all of the feature map entries have been correctly loaded
        for (const auto &entry : validFeatureIndex) {
            const auto itr = labelsToIndices.find(entry.first);
            CPPUNIT_ASSERT_MESSAGE("Each feature label from input should be present in feature index",
                itr != labelsToIndices.end());
            CPPUNIT_ASSERT_MESSAGE("Each feature index from input should be present in feature index",
                entry.second == itr->second);
        }
    }

    void TestLoadIndexWithDuplicateEntry() {
        // Seed the input stream with valid input
        stringstream inputStream;
        for (const auto &entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        // Duplicate first entry
        const auto itr = validFeatureIndex.begin();
        inputStream << validFeatureIndex.begin()->first << "\t" << itr->second << "\n";

        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    void TestLoadIndexWithDuplicateLabelOnly() {
        // Seed the input stream with valid input
        stringstream inputStream;
        for (const auto &entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        // Duplicate just the label used in the first entry
        inputStream << validFeatureIndex.begin()->first << "\t123\n";

        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    void TestLoadIndexWithMissingLabel() {
        stringstream inputStream;
        inputStream << "\t123\n";
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    void TestLoadIndexWithMissingLabelAndTab() {
        stringstream inputStream;
        inputStream << "123\n";
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    void TestLoadIndexWithExtraTab() {
        stringstream inputStream;
        inputStream << "110510\t123\t121017\n";
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    CPPUNIT_TEST_SUITE(TestNetCDFhelper);
    CPPUNIT_TEST(TestLoadIndexWithValidInput);
    CPPUNIT_TEST(TestLoadIndexWithDuplicateEntry);
    CPPUNIT_TEST(TestLoadIndexWithDuplicateLabelOnly);
    CPPUNIT_TEST(TestLoadIndexWithMissingLabel);
    CPPUNIT_TEST(TestLoadIndexWithMissingLabelAndTab);
    CPPUNIT_TEST(TestLoadIndexWithExtraTab);
    CPPUNIT_TEST_SUITE_END();
};

const map<string, unsigned int> TestNetCDFhelper::validFeatureIndex = {
    { "110510", 26743 },
    { "121019", 26740 },
    { "121017", 26739 },
    { "106401", 26736 },
    { "104307", 26734 }};
