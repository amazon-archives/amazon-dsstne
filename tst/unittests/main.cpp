#include <cstdlib>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>

// Test files
#include "TestNetCDFhelper.cpp"
#include "TestUtils.cpp"

//
// In order to write a new test case, create a Test<File>.cpp and write the
// test methods in that file. Include the cpp file in this file and add:
//
//    runner.addTest(Test<Class>::suite());
//
// Unit test file names have to start with 'Test'
//
int main()
{
    CppUnit::TextUi::TestRunner runner;
    runner.addTest(TestNetCDFhelper::suite());
    runner.addTest(TestUtils::suite());
    return runner.run() ? EXIT_SUCCESS : EXIT_FAILURE;
}
