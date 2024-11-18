//
// Created by victor on 11/17/24.
//

#include "testbase.h"

#include <cassert>
#include <cstdlib>
#include <sstream>


TestContext::TestContext(ostream &os) : os(os), passed(0), total(0),
                                        lastline(0), skip(false) {
    os << "line: ";
    os.width(65);
    os.setf(ios::left, ios::adjustfield);
    os << "description" << " result" << endl;
    os.width(78);
    os.fill('~');
    os << "~" << endl;
    os.fill(' ');
    os.setf(ios::right, ios::adjustfield);
}

void TestContext::desc(const string &msg, int line) {
    if ((lastline != 0) || ((msg[0] == '-') && skip))
        os << '\n';

    os.width(4);
    os << line << ": ";
    os.width(65);
    os.setf(ios::left, ios::adjustfield);
    os << msg << " ";
    os.setf(ios::right, ios::adjustfield);
    os.flush();

    lastline = line;
    skip = true;
}


void TestContext::check(bool test, int line) {
    if (!test)
        badlines.insert(line);
}


void TestContext::result() {
    assert(lastline != 0);

    // See if we haven't added any more values to the badlines collection
    auto iter = badlines.lower_bound(lastline);
    if (iter == badlines.end()) {
        os << "ok" << '\n';
        passed++;
    } else {
        os << "ERROR\n";

        while (iter != badlines.end()) {
            os << "\tFailure detected on line " << *iter << '\n';
            iter++;
        }
    }

    total++;
    lastline = 0;
}

TestContext::~TestContext() {
    os << "\nPassed " << passed << "/" << total << " tests.\n\n";

    if (badlines.size() > 2) {
        os << "We recommend that you try fixing the topmost failure and then"
                " re-test.\n"
                << "You may find that a single fix will resolve many failures.\n";
    }
}

bool TestContext::ok() const {
    return passed == total;
}
