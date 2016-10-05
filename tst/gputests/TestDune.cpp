// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
// STL
#include <string>

<<<<<<< Updated upstream
#include "TestGpu.cpp"
<<<<<<< Updated upstream
#include "TestSort.cpp"
=======
=======
#include "TestSort.cpp"
>>>>>>> Stashed changes
>>>>>>> Stashed changes

/**
 * In order to write a new test case, create a Test<File>.cpp and write the test
 * methods in that file. Include the cpp file in this file and also 
 *
 * add runner.addTest(Test<Class>::suite());
 * Unit test file name has to start with Test
 *
 */

int main() {
    CppUnit::TextUi::TestRunner runner;
<<<<<<< Updated upstream
    runner.addTest(TestGpu::suite());
<<<<<<< Updated upstream
    runner.addTest(TestSort::suite());
=======
=======
    runner.addTest(TestSort::suite());
>>>>>>> Stashed changes
>>>>>>> Stashed changes
    return !runner.run();
}
