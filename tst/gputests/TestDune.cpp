// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
// STL
#include <string>

#include "TestSort.cpp"
#include "TestActivationFunctions.cpp"
#include "TestCostFunctions.cpp"

/**
 * In order to write a new test case, create a Test<File>.cpp and write the test
 * methods in that file. Include the cpp file in this file and also 
 *
 * add runner.addTest(Test<Class>::suite());
 * Unit test file name has to start with Test
 *
 */

int main() {
    getGpu().Startup(0, NULL);
    getGpu().SetRandomSeed(12345);
    getGpu().CopyConstants();
    CppUnit::TextUi::TestRunner runner;
    runner.addTest(TestSort::suite());
    runner.addTest(TestActivationFunctions::suite());
    runner.addTest(TestCostFunctions::suite());
    const bool result = runner.run();
    getGpu().Shutdown();
    return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
