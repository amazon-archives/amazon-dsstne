// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
#include "cppunit/TestAssert.h"
// STL
#include <string>

#include "Utils.h"

//----------------------------------------------------------------------------
class TestUtils : public CppUnit::TestFixture
{
public:             // Interface
    void            TestIsNetCDFfile()
    {
        bool result=isNetCDFfile(std::string("network.nc"));
        CPPUNIT_ASSERT(1==result);     
        result=isNetCDFfile(std::string("network.nic"));
        CPPUNIT_ASSERT(0==result);     
    }
    
    
public:             // Boilerplate
    
    CPPUNIT_TEST_SUITE(TestUtils);
    CPPUNIT_TEST(TestIsNetCDFfile);
    CPPUNIT_TEST_SUITE_END();
    
};
