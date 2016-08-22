#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include "Utils.h"

class TestUtils : public CppUnit::TestFixture
{
public:
    void TestIsNetCDFfile()
    {
        bool result = isNetCDFfile("network.nc");
        CPPUNIT_ASSERT(result);

        result = isNetCDFfile("network.nic");
        CPPUNIT_ASSERT(!result);
    }
    
    CPPUNIT_TEST_SUITE(TestUtils);
    CPPUNIT_TEST(TestIsNetCDFfile);
    CPPUNIT_TEST_SUITE_END();
};
